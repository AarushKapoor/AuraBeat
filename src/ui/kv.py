# src/ui/kv.py
KV = r"""
#:kivy 2.3.0
#:import dp kivy.metrics.dp

<RootView>:
    orientation: "horizontal"
    spacing: dp(6)
    padding: dp(6)

    # LEFT: Stage (video + gear + quick menu + HUD)
    FloatLayout:
        id: stage
        size_hint_x: 0.38
        canvas.before:
            Color:
                rgba: 0.05, 0.06, 0.08, 1
            Rectangle:
                pos: self.pos
                size: self.size

        CircleButton:
            id: gear
            text: "⚙"
            size_hint: None, None
            size: dp(42), dp(42)
            pos: root.x + dp(8), root.top - dp(8) - self.height
            on_release: app.toggle_quick_menu()

        VideoFeed:
            id: video
            size_hint: None, None
            width: dp(420)
            height: self.width * 9/16
            pos_hint: {"top": 1}
            x: root.ids.gear.right + dp(8)
            fit_mode: "contain"

        GestureHUD:
            id: hud
            size_hint: None, None
            width: root.ids.video.width
            height: dp(110)
            x: root.ids.video.x
            y: root.ids.video.y - self.height - dp(4)

        QuickMenu:
            id: quickmenu
            size_hint: None, None
            size: dp(260), dp(260)
            pos: root.ids.gear.right + dp(10), root.ids.gear.top - self.height - dp(4)
            visible: False

    # MIDDLE: Air keyboard overlay (10 dots)
    AirOverlayPanel:
        id: overlay
        size_hint_x: 0.37

    # RIGHT: Piano roll
    PianoRollPanel:
        id: roll
        size_hint_x: 0.25
        keyboard_height_ratio: 0.18
        show_chevrons: True
"""