package proc

import (
	"bufio"
	"context"
	"fmt"
	"os/exec"
	"syscall"
	"testing"
	"time"
)

func alive(pid int) bool { return syscall.Kill(pid, 0) == nil }

// TestKillGroupKillsWholeTree verifies KillGroup reaps the child a CLI wrapper
// spawned, not just the wrapper itself — the orphaned-native-binary leak.
// Mirrors `codex`/`claude` being a node wrapper that execs a native child.
func TestKillGroupKillsWholeTree(t *testing.T) {
	// sh forks a `sleep` child and prints its PID, then waits. Killing only
	// sh would orphan the sleep; KillGroup must take down the whole group.
	cmd := exec.Command("sh", "-c", "sleep 30 & echo $!; wait")
	SetNewProcessGroup(cmd)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatalf("stdout pipe: %v", err)
	}
	if err := cmd.Start(); err != nil {
		t.Fatalf("start (need sh+sleep): %v", err)
	}

	var childPID int
	if _, err := fmt.Fscan(bufio.NewReader(stdout), &childPID); err != nil {
		t.Fatalf("read child pid: %v", err)
	}
	if childPID <= 0 {
		t.Fatalf("bad child pid: %d", childPID)
	}
	// Child must be alive before we kill the group.
	if err := syscall.Kill(childPID, 0); err != nil {
		t.Fatalf("child %d not alive pre-kill: %v", childPID, err)
	}

	KillGroup(cmd)
	_ = cmd.Wait()

	// The child sleep must be gone shortly after (no orphan).
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if err := syscall.Kill(childPID, 0); err != nil {
			return // child reaped -> success
		}
		time.Sleep(20 * time.Millisecond)
	}
	// best-effort cleanup so a failure doesn't leak the sleep
	_ = syscall.Kill(childPID, syscall.SIGKILL)
	t.Errorf("child sleep pid=%d still alive after KillGroup (orphan leak)", childPID)
}

// TestDrainLinesUnblocksBlockedProducer verifies DrainLines lets a producer
// blocked on `lines <- ...` (consumer stopped reading) run to completion —
// the mechanism that lets the real producer reach cmd.Wait() and avoid a
// <defunct> zombie.
func TestDrainLinesUnblocksBlockedProducer(t *testing.T) {
	lines := make(chan string, 2) // small buffer so it fills quickly
	finished := make(chan struct{})

	go func() {
		for i := 0; i < 10; i++ {
			lines <- "x" // blocks once buffer fills and nobody reads
		}
		close(lines)
		close(finished)
	}()

	// Let the producer fill the buffer and block.
	time.Sleep(50 * time.Millisecond)
	select {
	case <-finished:
		t.Fatal("producer finished without draining — test setup wrong")
	default:
	}

	DrainLines(lines)

	select {
	case <-finished:
		// success: drain unblocked the producer
	case <-time.After(2 * time.Second):
		t.Error("producer still blocked after DrainLines (zombie risk remains)")
	}
}

// TestKillGroupNilSafe ensures KillGroup is a no-op on nil/unstarted cmd.
func TestKillGroupNilSafe(t *testing.T) {
	KillGroup(nil)
	KillGroup(&exec.Cmd{}) // no Process
}

// TestWatchDisconnectKillsOnCancel: client disconnect (ctx cancel) must kill.
func TestWatchDisconnectKillsOnCancel(t *testing.T) {
	cmd := exec.Command("sh", "-c", "sleep 30")
	SetNewProcessGroup(cmd)
	if err := cmd.Start(); err != nil {
		t.Fatalf("start: %v", err)
	}
	pid := cmd.Process.Pid
	lines := make(chan string)
	close(lines) // empty: DrainLines returns immediately

	ctx, cancel := context.WithCancel(context.Background())
	stop := WatchDisconnect(ctx, func() *exec.Cmd { return cmd }, lines)
	defer stop()

	cancel() // simulate client disconnect

	// cmd.Wait() blocks until the WatchDisconnect goroutine kills the group;
	// a killed `sleep 30` returns a non-nil (signalled) error promptly. Using
	// Wait avoids the zombie pitfall where kill(pid,0) still reports a reaped-
	// pending process as "alive".
	waitErr := make(chan error, 1)
	go func() { waitErr <- cmd.Wait() }()
	select {
	case err := <-waitErr:
		if err == nil {
			t.Error("process exited normally; WatchDisconnect did not kill")
		}
	case <-time.After(3 * time.Second):
		_ = syscall.Kill(-pid, syscall.SIGKILL)
		t.Error("WatchDisconnect did not kill within 3s")
	}
}

// TestWatchDisconnectNoKillAfterNormalStop: stop() before cancel (normal
// completion) must NOT kill — guards the Wait-then-kill PID-reuse hazard.
func TestWatchDisconnectNoKillAfterNormalStop(t *testing.T) {
	cmd := exec.Command("sh", "-c", "sleep 30")
	SetNewProcessGroup(cmd)
	if err := cmd.Start(); err != nil {
		t.Fatalf("start: %v", err)
	}
	pid := cmd.Process.Pid
	defer func() { _ = syscall.Kill(-pid, syscall.SIGKILL); _ = cmd.Wait() }()

	lines := make(chan string)
	close(lines)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	stop := WatchDisconnect(ctx, func() *exec.Cmd { return cmd }, lines)
	stop()   // handler completed normally, before any cancel
	cancel() // ctx cancels after; goroutine already exited via done -> no kill

	time.Sleep(200 * time.Millisecond)
	if !alive(pid) {
		t.Error("WatchDisconnect killed after normal stop (Wait-then-kill hazard not guarded)")
	}
}
