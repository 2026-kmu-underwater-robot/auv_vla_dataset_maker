"""Exercise file and process boundaries without ROS or a desktop."""
import json
import sys
import threading

import pytest

from dashboard.backend import ManagedProcess, collector_command, export_command, scan_episodes


def test_scan_handles_complete_active_and_damaged_episodes(tmp_path):
    saved = tmp_path / 'episode_000001'
    saved.mkdir()
    (saved / 'manifest.json').write_text(json.dumps(dict(frames=25, fps=10,
        success=False, task='Recover.', termination_reason='node_shutdown')))
    damaged = tmp_path / 'episode_000002'
    damaged.mkdir()
    (damaged / 'manifest.json').write_text('{broken')
    active = tmp_path / '.recording_episode_000003'
    active.mkdir()
    rows, recording, errors = scan_episodes(tmp_path)
    assert rows[0]['seconds'] == 2.5
    assert rows[0]['success'] is False
    assert rows[0]['reason'] == 'node_shutdown'
    assert recording == [str(active)]
    assert len(errors) == 1
    assert scan_episodes(tmp_path / 'missing') == ([], [], [])


def test_commands_keep_paths_and_task_identity_as_literal_arguments(tmp_path):
    config = tmp_path / 'config with spaces.yaml'
    config.write_text('')
    args = collector_command(config, tmp_path / 'a;$(echo hello)', True)
    assert args[args.index('--params-file') + 1] == str(config)
    assert 'use_sim_time:=true' in args
    assert f'dataset_root:={tmp_path / "a;$(echo hello)"}' in args
    command = export_command('/env with spaces/bin/python', tmp_path,
                             tmp_path / 'new output', 'user/dataset')
    assert command[0] == '/env with spaces/bin/python'
    assert command[-2:] == ['--repo-id', 'user/dataset']
    with pytest.raises(ValueError, match='새 폴더'):
        export_command(sys.executable, tmp_path, tmp_path, 'user/data')
    with pytest.raises(ValueError, match='설정 파일'):
        collector_command(tmp_path / 'missing', tmp_path, False)
    with pytest.raises(ValueError, match='형식'):
        export_command(sys.executable, tmp_path, tmp_path / 'new', 'bad-id')


def test_process_streams_output_and_interrupts_cleanly(tmp_path):
    ready = threading.Event()
    finished = threading.Event()
    lines = []
    def log(line):
        lines.append(line)
        if line == 'ready':
            ready.set()
        if '프로세스 종료' in line:
            finished.set()
    script = tmp_path / 'child.py'
    script.write_text('import signal, time\n'
                      'signal.signal(signal.SIGINT, lambda *args: exit(0))\n'
                      'print("ready", flush=True)\n'
                      'while True: time.sleep(0.05)\n')
    process = ManagedProcess(log)
    process.start([sys.executable, str(script)])
    try:
        assert ready.wait(5)
        with pytest.raises(RuntimeError, match='이미 실행'):
            process.start([sys.executable, str(script)])
        process.stop()
        assert finished.wait(5)
        assert not process.running
        assert process.process.returncode == 0
    finally:
        process.stop()
        if process.running:
            process.process.kill()
            process.process.wait()
