# src/ui/widgets/docks.py
from __future__ import annotations

from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.properties import NumericProperty, ListProperty
from kivy.metrics import dp
from kivy.app import App
from .controls import PillButton, CircleIconButton
from .transport_controls import TransportControls


# ---------------------------------------------------------------------
# UpperDock
# ---------------------------------------------------------------------
class UpperDock(Widget):
    margin_dp = NumericProperty(dp(10))
    spacing_dp = NumericProperty(dp(8))
    pad_dp = ListProperty([dp(8), dp(8), dp(8), dp(8)])

    pill_height_dp = NumericProperty(dp(44))
    record_diameter_dp = NumericProperty(dp(56))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._row = BoxLayout(
            orientation="horizontal",
            spacing=self.spacing_dp,
            padding=self.pad_dp,
            size_hint=(None, None),
            height=self.pill_height_dp,
            width=0
        )
        self.add_widget(self._row)

        self._rec_slot = AnchorLayout(
            anchor_x="left",
            anchor_y="center",
            size_hint=(None, 1),
            width=self.record_diameter_dp
        )

        self._transport_slot = AnchorLayout(
            anchor_x="left",
            anchor_y="center",
            size_hint=(None, 1),
            width=dp(200)
        )

        self._inst_slot = AnchorLayout(
            anchor_x="left",
            anchor_y="center",
            size_hint=(None, 1),
            width=dp(180)
        )

        self._layer_slot = AnchorLayout(
            anchor_x="left",
            anchor_y="center",
            size_hint=(None, 1),
            width=dp(130)
        )

        self._row.add_widget(self._rec_slot)
        self._row.add_widget(self._transport_slot)
        self._row.add_widget(self._inst_slot)
        self._row.add_widget(self._layer_slot)

        # Record button
        self.btn_record = CircleIconButton(size_hint=(None, None))
        self.btn_record.size = (self.record_diameter_dp, self.record_diameter_dp)
        self._rec_slot.add_widget(self.btn_record)
        self.btn_record.bind(on_release=self._on_record_pressed)

        # Transport controls
        self.transport = TransportControls()
        self.transport.height = self.pill_height_dp
        self.transport.width = dp(200)
        self._transport_slot.add_widget(self.transport)

        app = App.get_running_app()
        self.transport.on_play   = lambda: app.toggle_playback() if app else None
        self.transport.on_pause  = lambda: app.playback.pause() if app and app.playback else None
        self.transport.on_back   = lambda: app.playback.stop() if app and app.playback else None
        self.transport.on_forward = lambda: app.playback.stop() if app and app.playback else None
        self.transport.on_loop   = lambda: app.playback.enable_loop(app.playback.current_time + 4) if app and app.playback else None

        # Instrument button
        self.btn_instrument = PillButton(text="Instrument: Piano", size_hint=(None, None))
        self.btn_instrument.height = self.pill_height_dp
        self.btn_instrument.width = self._inst_slot.width
        self._inst_slot.add_widget(self.btn_instrument)

        # Layer button
        self.btn_layer = PillButton(text="Layer: 1", size_hint=(None, None))
        self.btn_layer.height = self.pill_height_dp
        self.btn_layer.width = self._layer_slot.width
        self._layer_slot.add_widget(self.btn_layer)

        self._recompute_row_width()

        self.bind(
            pos=self._sync, size=self._sync,
            spacing_dp=self._rebuild_layout,
            pad_dp=self._rebuild_layout,
            pill_height_dp=self._sync_sizes,
            record_diameter_dp=self._sync_sizes
        )

    def _sync_sizes(self, *_):
        self._row.height = self.pill_height_dp
        self.btn_instrument.height = self.pill_height_dp
        self.btn_layer.height = self.pill_height_dp
        self._rec_slot.width = self.record_diameter_dp
        self.btn_record.size = (self.record_diameter_dp, self.record_diameter_dp)
        self._inst_slot.width = self.btn_instrument.width
        self._layer_slot.width = self.btn_layer.width
        self._recompute_row_width()

    def _rebuild_layout(self, *_):
        self._row.spacing = self.spacing_dp
        self._row.padding = self.pad_dp
        self._recompute_row_width()

    def _recompute_row_width(self):
        total = 0
        for ch in self._row.children:
            total += ch.width
        if len(self._row.children) > 1:
            total += self._row.spacing * (len(self._row.children) - 1)
        self._row.width = total

    def _sync(self, *_):
        self._row.pos = (
            self.x + self.margin_dp,
            self.top - self._row.height - self.margin_dp
        )

    def _on_record_pressed(self, *args):
        app = App.get_running_app()
        if not app:
            return
        app.toggle_record()
        self.btn_record.active = not self.btn_record.active


# ---------------------------------------------------------------------
# LowerDock
# ---------------------------------------------------------------------
class LowerDock(Widget):
    margin_dp = NumericProperty(dp(10))
    spacing_dp = NumericProperty(dp(8))
    pad_dp = ListProperty([dp(8), dp(8), dp(8), dp(8)])

    pill_height_dp = NumericProperty(dp(44))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._container = BoxLayout(
            orientation="horizontal",
            padding=self.pad_dp,
            size_hint=(1, None),
            height=self.pill_height_dp
        )
        self._row = BoxLayout(
            orientation="horizontal",
            spacing=self.spacing_dp,
            size_hint=(None, 1),
            width=0
        )
        self._container.add_widget(self._row)
        self.add_widget(self._container)

        self._muted = False

        self.btn_mute = PillButton(text="Mute", size_hint=(None, None))
        self.btn_mute.height = self.pill_height_dp
        self.btn_mute.width = dp(110)
        self.btn_mute.bind(on_release=self._on_mute_pressed)

        self.btn_controls = PillButton(text="Controls: 10-key mapping", size_hint=(None, None))
        self.btn_controls.height = self.pill_height_dp
        self.btn_controls.width = dp(230)

        self._row.add_widget(self.btn_mute)
        self._row.add_widget(self.btn_controls)
        self._recompute_row_width()

        self.bind(
            pos=self._sync, size=self._sync,
            spacing_dp=self._rebuild_layout,
            pad_dp=self._rebuild_layout,
            pill_height_dp=self._sync_sizes
        )

    def _on_mute_pressed(self, *args):
        app = App.get_running_app()
        if not app:
            return
        self._muted = not self._muted
        app.set_muted(self._muted)
        if self._muted:
            self.btn_mute.text = "Unmute"
            self.btn_mute.color = (1, 0.3, 0.3, 1)
        else:
            self.btn_mute.text = "Mute"
            self.btn_mute.color = (0.92, 0.95, 1, 1)

    def _sync_sizes(self, *_):
        self._container.height = self.pill_height_dp
        self.btn_mute.height = self.pill_height_dp
        self.btn_controls.height = self.pill_height_dp
        self._recompute_row_width()

    def _rebuild_layout(self, *_):
        self._container.padding = self.pad_dp
        self._row.spacing = self.spacing_dp
        self._recompute_row_width()

    def _recompute_row_width(self):
        total = 0
        for ch in self._row.children:
            total += ch.width
        if len(self._row.children) > 1:
            total += self._row.spacing * (len(self._row.children) - 1)
        self._row.width = total

    def _sync(self, *_):
        self._container.pos = (
            self.x + self.margin_dp,
            self.y + self.margin_dp
        )
        self._container.width = self.width - 2 * self.margin_dp
        self._container.height = self.pill_height_dp