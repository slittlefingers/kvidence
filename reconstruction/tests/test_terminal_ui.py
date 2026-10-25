"""Terminal controls must preserve tasks and consume only complete event records."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
spec = importlib.util.spec_from_file_location('terminal_ui', APP / 'terminal.py')
ui = importlib.util.module_from_spec(spec); spec.loader.exec_module(ui)


class TerminalTests(unittest.TestCase):
    def test_partial_event_is_not_lost(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.jsonl'
            start = dict(type='start', config={}, planned_positions=1, target_positions=[9])
            finish = dict(type='finish', result=dict(status='complete', online_seconds=2))
            encoded = json.dumps(finish)
            path.write_text(json.dumps(start) + '\n' + encoded[:10])
            reader = ui.EventReader(path); reader.poll()
            self.assertEqual(reader.state.data['status'], 'running')
            with path.open('a') as stream:stream.write(encoded[10:] + '\n')
            reader.poll()
            self.assertEqual(reader.state.data['status'], 'complete')
            offset = reader.offset; reader.poll()
            self.assertEqual(reader.offset, offset)

    def test_demo_cannot_change_live_collection(self):
        terminal = ui.Terminal(demo=True)
        try:
            with patch.object(ui, 'api') as api:
                for command in ('/pause', '/resume', '/use fixed991', '/run --output x'):
                    terminal.execute(command)
                api.assert_not_called()
                self.assertFalse(terminal.pending)
        finally:terminal.pool.shutdown()

    def test_run_blocked_for_active_or_pending_collection(self):
        terminal = ui.Terminal()
        try:
            for state in (dict(active=True, mode='running'), dict(active=False, mode='waiting')):
                with patch.object(ui, 'api', return_value=state):
                    with self.assertRaisesRegex(ValueError, '/pause'):
                        terminal.prepare_run(['--output', 'new_run'])
        finally:terminal.pool.shutdown()

    def test_quit_does_not_stop_recovery(self):
        terminal = ui.Terminal(); terminal.process = Mock()
        try:
            with patch.object(ui.os, 'killpg') as kill:
                terminal.execute('/quit')
                self.assertFalse(terminal.running); kill.assert_not_called()
                terminal.process.terminate.assert_not_called()
        finally:terminal.pool.shutdown()

    def test_fixed_case_commands_do_not_load_an_agent(self):
        from case_commands import CaseCommands
        from casework import CaseStore
        terminal=ui.Terminal();terminal.next_poll=float('inf')
        try:
            with tempfile.TemporaryDirectory() as directory:
                terminal.case_commands=CaseCommands(CaseStore(Path(directory)))
                terminal.execute('/case new case1')
                terminal.pending[0][1].result(timeout=2)
                terminal.tick()
                self.assertIsNone(terminal.agent)
                self.assertEqual(terminal.case_name,'case1')
                self.assertTrue((Path(directory)/'case1/metadata.json').is_file())
        finally:terminal.pool.shutdown()

    def test_recovery_respects_phone_cooling(self):
        terminal = ui.Terminal()
        try:
            with patch.object(ui,'api',return_value=dict(active=False,mode='paused',cooldown=120)):
                with self.assertRaisesRegex(ValueError,'cooling'):
                    terminal.prepare_run(['--output','new_run'])
        finally:terminal.pool.shutdown()

    def test_launcher_leaves_output_creation_to_recovery(self):
        terminal = ui.Terminal()
        try:
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / 'new run'
                def launch(argv, **kwargs):
                    self.assertFalse(output.exists())
                    self.assertEqual(argv[argv.index('--output')+1],str(output.resolve()))
                    self.assertIn('--live-events',argv)
                    return Mock(pid=123)
                with patch.object(ui.subprocess,'Popen',side_effect=launch):
                    terminal.start_run(['--observation','/some/observation','--output',str(output)])
                self.assertTrue(terminal.log.is_file())
                self.assertFalse(output.exists())
        finally:terminal.pool.shutdown()

    def test_bracketed_paste_does_not_execute_newline(self):
        terminal = ui.Terminal()
        try:
            screen = Mock(); screen.get_wch.side_effect = list('[200~')
            terminal.key(screen, '\x1b')
            with patch.object(terminal, 'execute') as execute:
                for char in '/pause\n':terminal.key(screen, char)
                execute.assert_not_called()
            self.assertEqual(terminal.input, '/pause ')
            screen.get_wch.side_effect = list('[201~'); terminal.key(screen, '\x1b')
            self.assertFalse(terminal.pasting)
        finally:terminal.pool.shutdown()

    def test_help_starts_at_top_and_all_commands_are_reachable(self):
        class Screen:
            def getmaxyx(self):return 24,80
            def erase(self):self.text=[]
            def addstr(self,y,x,text,*args):self.text.append(text)
            def refresh(self):pass
        terminal=ui.Terminal(demo=True);screen=Screen()
        try:
            terminal.execute('/help')
            self.assertEqual(terminal.help_offset,0)
            seen=[]
            with patch.object(ui.curses,'color_pair',return_value=0):
                terminal.paint(screen)
                self.assertIn('/status',screen.text)
                for _ in range(12):
                    seen.extend(screen.text)
                    previous=terminal.help_offset
                    terminal.key(screen,ui.curses.KEY_NPAGE);terminal.paint(screen)
                    if terminal.help_offset==previous:break
            self.assertIn('/case new NAME',seen)
            self.assertIn('/case run [NAME]',seen)
            terminal.key(screen,'q');self.assertIsNone(terminal.help_view)
            terminal.execute('/case help');self.assertEqual(terminal.help_offset,0)
        finally:terminal.pool.shutdown()

    def test_chinese_width_and_control_characters(self):
        self.assertEqual(ui.cells('恢复abc'), 7)
        for line in ui.wrap('图片恢复 text', 6):self.assertLessEqual(ui.cells(line), 6)
        self.assertNotIn('\x1b', ui.clean('\x1b[31mhello'))

    def test_layout_fits_and_displays_english(self):
        class Screen:
            def __init__(self, height, width):
                self.height = height; self.width = width; self.draws = []
            def getmaxyx(self):return self.height, self.width
            def erase(self):self.draws.clear()
            def addstr(self, y, x, text, *args):
                assert 0 <= y < self.height
                assert x + ui.cells(text) < self.width
                self.draws.append((y, text))
            def move(self, y, x):
                assert 0 <= y < self.height and 0 <= x < self.width
            def refresh(self):pass
        terminal = ui.Terminal(demo=True)
        try:
            terminal.tick()
            with patch.object(ui.curses, 'color_pair', return_value=0):
                for h, w in ((16, 54), (24, 80), (34, 120)):
                    screen = Screen(h, w); terminal.paint(screen)
                    rendered = '\n'.join(text for _, text in screen.draws)
                    self.assertIn('KVidence', rendered)
                    self.assertIn('images', rendered)
                    self.assertFalse(any('\u4e00' <= c <= '\u9fff' for c in rendered))
                terminal.execute('/help')
                terminal.paint(screen)
                self.assertFalse(any('\u4e00' <= c <= '\u9fff' for _, text in screen.draws for c in text))
        finally:terminal.pool.shutdown()


if __name__ == '__main__':unittest.main()
