# src/ui/kv.py
KV = r"""
#:kivy 2.3.0
#:import dp kivy.metrics.dp

<RootView>:
    canvas.before:
        Color:
            rgba: (0x1A/255., 0x2B/255., 0x43/255., 1)
        Rectangle:
            pos: self.pos
            size: self.size

    RelativeLayout:
        id: root_stage
        size_hint: 1, 1

        BoxLayout:
            id: main_columns
            orientation: "horizontal"
            spacing: dp(8)
            padding: 0, 0, 0, 0
            size_hint: 1, 1

            # ------------------ Column 1 (Left) ---------------------
            RelativeLayout:
                id: left_stack
                size_hint_x: 0.2

                LeftOptionsPanel:
                    id: left_bg
                    size_hint: 1, 1
                    pos: self.parent.pos

                BoxLayout:
                    id: col_left
                    orientation: "vertical"
                    spacing: dp(8)
                    padding: 0, 0, 0, 0
                    size_hint: 1, 1

                    VideoFeed:
                        id: video
                        size_hint: 1, None
                        height: max(dp(220), self.width * 9/16)
                        fit_mode: "contain"
                        draw_border: False

                    GestureHUD:
                        id: hud
                        size_hint: 1, None
                        height: dp(110)

                    Widget:
                        size_hint_y: 1

            # ------------------ Column 2 (Middle) ---------------------
            AirOverlayPanel:
                id: overlay
                size_hint_x: 0.6

            # ------------------ Column 3 (Right) ----------------------
            PianoRollPanel:
                id: roll
                size_hint_x: 0.2
                keyboard_height_ratio: 0.18
                show_chevrons: False


        
        UpperDock:
            x: overlay.parent.x + overlay.x
            y: root.top - self.height - dp(10)
            size_hint: None, None
            width: overlay.width
            height: self.pill_height_dp  # optional: or leave a fixed dp(...) if you prefer
        LowerDock:
            x: overlay.parent.x + overlay.x
            y: root.y + dp(10)
            size_hint: None, None
            width: overlay.width
            height: self.pill_height_dp

"""