// Package proc provides process-group lifecycle helpers shared by the
// relay-claude and relay-codex services.
//
// Both relays spawn a CLI (`claude` / `codex`) which is itself a Node.js
// wrapper that execs a native binary. Killing only cmd.Process kills the Node
// wrapper but lets the native child reparent to init and keep running (an
// orphan); and if the producer goroutine is blocked on `lines <- ...` because
// the consumer stopped reading, cmd.Wait() never runs and the killed wrapper
// becomes a <defunct> zombie. These helpers fix both:
//
//   - SetNewProcessGroup: start the child in its own process group so the
//     whole tree (wrapper + native child) can be signalled at once.
//   - KillGroup: signal the entire group via the negative PID, so no orphan
//     native binary is left behind.
//   - DrainLines: drain the output channel so the producer goroutine unblocks,
//     finishes its scan loop and reaches cmd.Wait(), reaping the wrapper.
package proc

import (
	"context"
	"log"
	"os/exec"
	"sync"
	"syscall"
)

// SetNewProcessGroup makes the child start in its own process group so the
// whole subtree can be killed with a single signal to the negative PID.
// Must be called before cmd.Start().
func SetNewProcessGroup(cmd *exec.Cmd) {
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.Setpgid = true
}

// KillGroup SIGKILLs the entire process group of cmd (relies on
// SetNewProcessGroup having been called). This reaps the Node wrapper *and*
// the native child it spawned, preventing orphaned native binaries.
// Idempotent and safe to call when the process has already exited.
func KillGroup(cmd *exec.Cmd) {
	if cmd == nil || cmd.Process == nil {
		return
	}
	pid := cmd.Process.Pid
	if pid <= 0 {
		return
	}
	// Negative PID targets the whole process group. Fall back to a single
	// process kill if the group signal fails (e.g. Setpgid didn't take).
	if err := syscall.Kill(-pid, syscall.SIGKILL); err != nil {
		_ = cmd.Process.Kill()
	}
}

// InterruptGroup SIGINTs the entire process group of cmd, asking the CLI to
// abort gracefully. Unlike KillGroup, a SIGINTed `claude` still emits a final
// `result` event (subtype error_during_execution) whose modelUsage carries the
// interrupted turn's real token counts — callers use this to account for
// aborted turns before falling back to KillGroup.
func InterruptGroup(cmd *exec.Cmd) {
	if cmd == nil || cmd.Process == nil {
		return
	}
	pid := cmd.Process.Pid
	if pid <= 0 {
		return
	}
	if err := syscall.Kill(-pid, syscall.SIGINT); err != nil {
		_ = cmd.Process.Signal(syscall.SIGINT)
	}
}

// DrainLines drains the producer's output channel in the background so a
// goroutine blocked on `lines <- ...` (consumer stopped reading) unblocks,
// completes its scan loop and runs cmd.Wait(), preventing a <defunct> zombie.
// The drain goroutine exits when the channel is closed (which the producer
// does after cmd.Wait()).
func DrainLines(lines <-chan string) {
	go func() {
		for range lines { //nolint:revive // intentional drain
		}
	}()
}

// WatchDisconnect spawns a goroutine that, when ctx is cancelled (the client
// dropped the connection), kills the child's process group and drains lines.
//
// It returns a stop func the handler MUST defer. Calling stop on normal
// completion makes the goroutine exit WITHOUT killing — essential because by
// then the child has already been reaped by cmd.Wait() and its PID may have
// been reused, so a bare KillGroup could SIGKILL an unrelated process group.
//
// getCmd is a closure so callers whose *exec.Cmd is swapped at runtime
// (claude's resume-retry) always signal the current child.
func WatchDisconnect(ctx context.Context, getCmd func() *exec.Cmd, lines <-chan string) (stop func()) {
	done := make(chan struct{})
	go func() {
		select {
		case <-ctx.Done():
			cmd := getCmd()
			if cmd != nil && cmd.Process != nil {
				log.Printf("client disconnected, killing process group pid=%d", cmd.Process.Pid)
			}
			KillGroup(cmd)
			DrainLines(lines)
		case <-done:
			// handler completed normally; child already exited and was Wait()ed.
		}
	}()
	var once sync.Once
	return func() { once.Do(func() { close(done) }) }
}
