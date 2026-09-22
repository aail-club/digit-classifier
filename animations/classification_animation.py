from manim import *


DIGIT_FEATURES = {
    0: [0, 1, 2, 5],
    1: [1, 3, 5],
    2: [0, 3, 4, 5],
    3: [0, 2, 3, 5],
    4: [1, 2, 3, 4],
    5: [0, 1, 2, 4],
    6: [0, 1, 2, 4, 5],
    7: [0, 3, 5],
    8: [0, 1, 2, 3, 4, 5],
    9: [0, 1, 2, 3, 5],
}


class DigitClassificationScene:
    digit = 0

    def construct(self):
        # ============================================================
        # LAYERS
        #
        # Symmetric around x = 0:
        #
        #   -3.45       -1.15       +1.15       +3.45
        #    pixels      strokes     features     outputs
        # ============================================================

        input_visual, input_nodes = self.make_long_layer(
            color=WHITE,
            visible_each_side=4,
        )
        input_visual.move_to(LEFT * 4)

        strokes_layer = self.make_layer(
            6,
            BLUE,
        ).move_to(LEFT * 1.15)

        hidden_layer = self.make_layer(
            7,
            TEAL,
        ).move_to(RIGHT * 1.15)

        output_nodes, output_labels = self.make_output_layer()
        output_nodes.move_to(RIGHT * 3.45)

        # Reposition labels after moving output nodes.
        for index, label in enumerate(output_labels):
            label.next_to(
                output_nodes[index],
                RIGHT,
                buff=0.16,
            )

        # ============================================================
        # LABELS
        # ============================================================

        input_label = Text(
            "pixels",
            font_size=36,
        ).next_to(
            input_visual,
            UP,
            buff=0.32,
        )

        strokes_label = Text(
            "strokes",
            font_size=36,
        ).next_to(
            strokes_layer,
            UP,
            buff=0.32,
        )

        features_label = Text(
            "features",
            font_size=36,
        ).next_to(
            hidden_layer,
            UP,
            buff=0.32,
        )

        outputs_label = Text(
            "outputs",
            font_size=36,
        ).next_to(
            output_nodes,
            UP,
            buff=0.32,
        )

        # ============================================================
        # CAMERA
        # ============================================================

        camera_label = Text(
            "camera",
            font_size=32,
        ).next_to(
            input_visual,
            LEFT,
            buff=1.15,  # more room for arrow
        )
        
        camera_arrow = Arrow(
            camera_label.get_right(),
            input_visual.get_left(),
            buff=0.12,
            color=YELLOW,
            stroke_width=5,
            max_tip_length_to_length_ratio=0.18,
        )

        # ============================================================
        # CONNECTIONS
        # ============================================================

        pixel_connections = self.connect_layers(
            input_nodes,
            strokes_layer,
        )

        highlighted_pixel_connections = (
            self.make_highlighted_pixel_connections(
                input_nodes,
                strokes_layer,
            )
        )

        stroke_connections = self.connect_layers(
            strokes_layer,
            hidden_layer,
        )

        output_connections_base = self.connect_layers(
            hidden_layer,
            output_nodes,
        )

        (
            feature_connections,
            highlighted_output_connections,
        ) = self.make_highlighted_connections(
            strokes_layer,
            hidden_layer,
            output_nodes,
        )

        # Base network behind highlighted paths / neurons.
        pixel_connections.set_z_index(-2)
        stroke_connections.set_z_index(-2)
        output_connections_base.set_z_index(-2)

        highlighted_pixel_connections.set_z_index(-1)
        feature_connections.set_z_index(-1)
        highlighted_output_connections.set_z_index(-1)

        # ============================================================
        # PREDICTION
        # ============================================================

        prediction = Text(
            f"Prediction: {self.digit}",
            font_size=60,
            weight=BOLD,
            color=GREEN,
        )

        network_bottom = min(
            input_visual.get_bottom()[1],
            strokes_layer.get_bottom()[1],
            hidden_layer.get_bottom()[1],
            output_nodes.get_bottom()[1],
        )

        prediction.move_to(
            [0, network_bottom - 0.55, 0]
        )

        # ============================================================
        # ACTIVE NODES
        # ============================================================

        active_inputs = DIGIT_FEATURES[self.digit]

        active_hidden = sorted(
            {
                (feature * 2 + self.digit) % len(hidden_layer)
                for feature in active_inputs
            }
        )

        active_hidden += [
            index
            for index in range(len(hidden_layer))
            if index not in active_hidden
        ][:2]

        active_hidden = active_hidden[:5]

        # ============================================================
        # ANIMATION
        # ============================================================

        self.play(
            Write(camera_label),
            run_time=1.5,
        )

        self.wait(0.4)

        # Camera -> pixels
        self.play(
            GrowArrow(camera_arrow),
            FadeIn(input_visual),
            Write(input_label),
            run_time=2.0,
        )

        self.wait(0.5)

        # Pixels -> strokes
        self.play(
            Create(pixel_connections),
            FadeIn(strokes_layer),
            Write(strokes_label),
            run_time=2.5,
        )

        self.wait(0.6)

        # Strokes -> features
        self.play(
            Create(stroke_connections),
            FadeIn(hidden_layer),
            Write(features_label),
            run_time=2.5,
        )

        self.wait(0.6)

        # Features -> outputs
        self.play(
            Create(output_connections_base),
            FadeIn(output_nodes),
            FadeIn(output_labels),
            Write(outputs_label),
            run_time=2.5,
        )

        self.wait(0.7)

        # ============================================================
        # PIXELS -> DETECTED STROKES
        # ============================================================

        self.wait(0.5)

        self.play(
            LaggedStart(
                *[
                    strokes_layer[index]
                    .animate
                    .set_fill(
                        BLUE,
                        opacity=0.95,
                    )
                    for index in active_inputs
                ],
                lag_ratio=0.14,
            ),
            Create(highlighted_pixel_connections),
            run_time=2.0,
        )

        self.wait(0.6)

        # ============================================================
        # STROKES -> FEATURES
        # ============================================================

        self.play(
            LaggedStart(
                *[
                    strokes_layer[index]
                    .animate
                    .set_fill(
                        YELLOW,
                        opacity=0.95,
                    )
                    for index in active_inputs
                ],
                lag_ratio=0.14,
            ),
            Create(feature_connections),
            run_time=2.0,
        )

        self.wait(0.7)

        # ============================================================
        # FEATURES -> OUTPUT
        # ============================================================

        self.play(
            LaggedStart(
                *[
                    hidden_layer[index]
                    .animate
                    .set_fill(
                        ORANGE,
                        opacity=0.9,
                    )
                    for index in active_hidden
                ],
                lag_ratio=0.14,
            ),
            Create(highlighted_output_connections),
            run_time=2.0,
        )

        self.wait(0.7)

        # ============================================================
        # FINAL PREDICTION
        # ============================================================

        self.play(
            output_nodes[self.digit]
            .animate
            .set_fill(
                GREEN,
                opacity=1.0,
            )
            .scale(1.22),

            output_labels[self.digit]
            .animate
            .set_color(GREEN)
            .scale(1.18),

            Write(prediction),

            run_time=1.8,
        )

        self.wait(2.0)

    # ================================================================
    # STANDARD LAYER
    # ================================================================

    def make_layer(self, count, color):
        nodes = VGroup(
            *[
                Circle(
                    radius=0.17,
                    color=color,
                    stroke_width=2.2,
                    fill_color=color,
                    fill_opacity=0.32,
                )
                for _ in range(count)
            ]
        )

        nodes.arrange(
            DOWN,
            buff=0.16,
        )

        return nodes

    # ================================================================
    # HIGHLIGHTED PIXEL -> STROKE CONNECTIONS
    # ================================================================

    def make_highlighted_pixel_connections(
        self,
        pixel_nodes,
        strokes_layer,
    ):
        lines = VGroup()

        active_strokes = DIGIT_FEATURES[self.digit]

        for stroke_index in active_strokes:
            pixel_indices = {
                (stroke_index + self.digit)
                % len(pixel_nodes),

                (stroke_index * 2 + self.digit)
                % len(pixel_nodes),

                (stroke_index * 3 + self.digit + 1)
                % len(pixel_nodes),
            }

            for pixel_index in pixel_indices:
                lines.add(
                    Line(
                        pixel_nodes[pixel_index].get_right(),
                        strokes_layer[stroke_index].get_left(),
                        color=BLUE,
                        stroke_width=4.2,
                        stroke_opacity=0.95,
                    )
                )

        return lines

    # ================================================================
    # ABBREVIATED 784-PIXEL LAYER
    # ================================================================

    def make_long_layer(
        self,
        color,
        visible_each_side=4,
    ):
        top_nodes = [
            Circle(
                radius=0.17,
                color=color,
                stroke_width=2.2,
                fill_color=color,
                fill_opacity=0.32,
            )
            for _ in range(visible_each_side)
        ]

        bottom_nodes = [
            Circle(
                radius=0.17,
                color=color,
                stroke_width=2.2,
                fill_color=color,
                fill_opacity=0.32,
            )
            for _ in range(visible_each_side)
        ]

        ellipsis = Text(
            "⋮",
            font_size=40,
            color=color,
        )

        visual = VGroup(
            *top_nodes,
            ellipsis,
            *bottom_nodes,
        )

        visual.arrange(
            DOWN,
            buff=0.16,
        )

        # Only actual circles are connection targets.
        nodes = VGroup(
            *top_nodes,
            *bottom_nodes,
        )

        return visual, nodes

    # ================================================================
    # OUTPUT LAYER
    # ================================================================

    def make_output_layer(self):
        nodes = self.make_layer(
            10,
            GREEN,
        )

        labels = VGroup(
            *[
                Text(
                    str(index),
                    font_size=25,
                ).next_to(
                    nodes[index],
                    RIGHT,
                    buff=0.16,
                )
                for index in range(10)
            ]
        )

        return nodes, labels

    # ================================================================
    # BASE CONNECTIONS
    # ================================================================

    def connect_layers(
        self,
        left_layer,
        right_layer,
    ):
        lines = VGroup()

        for left_node in left_layer:
            for right_node in right_layer:
                lines.add(
                    Line(
                        left_node.get_right(),
                        right_node.get_left(),
                        stroke_width=1.4,
                        color=GRAY,
                        stroke_opacity=0.38,
                    )
                )

        return lines

    # ================================================================
    # HIGHLIGHTED CONNECTIONS
    # ================================================================

    def make_highlighted_connections(
        self,
        input_layer,
        hidden_layer,
        output_layer,
    ):
        active_inputs = DIGIT_FEATURES[self.digit]

        feature_lines = VGroup()
        output_lines = VGroup()

        # ------------------------------------------------------------
        # Strokes -> hidden features
        # ------------------------------------------------------------

        for feature in active_inputs:
            targets = [
                (feature + self.digit)
                % len(hidden_layer),

                (feature * 2 + self.digit + 1)
                % len(hidden_layer),
            ]

            for target in targets:
                feature_lines.add(
                    Line(
                        input_layer[feature].get_right(),
                        hidden_layer[target].get_left(),
                        color=YELLOW,
                        stroke_width=4.2,
                        stroke_opacity=0.95,
                    )
                )

        # ------------------------------------------------------------
        # Hidden features -> predicted output
        # ------------------------------------------------------------

        hidden_targets = sorted(
            {
                (feature * 2 + self.digit)
                % len(hidden_layer)
                for feature in active_inputs
            }
        )[:4]

        for source in hidden_targets:
            output_lines.add(
                Line(
                    hidden_layer[source].get_right(),
                    output_layer[self.digit].get_left(),
                    color=GREEN,
                    stroke_width=4.5,
                    stroke_opacity=0.95,
                )
            )

        # ------------------------------------------------------------
        # Weak competing outputs
        # ------------------------------------------------------------

        nearby_digits = {
            max(0, self.digit - 1),
            min(9, self.digit + 1),
        } - {self.digit}

        for nearby in nearby_digits:
            source = (
                nearby + self.digit
            ) % len(hidden_layer)

            output_lines.add(
                Line(
                    hidden_layer[source].get_right(),
                    output_layer[nearby].get_left(),
                    color=ORANGE,
                    stroke_width=2.3,
                    stroke_opacity=0.55,
                )
            )

        return feature_lines, output_lines


# ====================================================================
# SCENES
# ====================================================================


class ClassificationAnimation0(DigitClassificationScene, Scene):
    digit = 0


class ClassificationAnimation1(DigitClassificationScene, Scene):
    digit = 1


class ClassificationAnimation2(DigitClassificationScene, Scene):
    digit = 2


class ClassificationAnimation3(DigitClassificationScene, Scene):
    digit = 3


class ClassificationAnimation4(DigitClassificationScene, Scene):
    digit = 4


class ClassificationAnimation5(DigitClassificationScene, Scene):
    digit = 5


class ClassificationAnimation6(DigitClassificationScene, Scene):
    digit = 6


class ClassificationAnimation7(DigitClassificationScene, Scene):
    digit = 7


class ClassificationAnimation8(DigitClassificationScene, Scene):
    digit = 8


class ClassificationAnimation9(DigitClassificationScene, Scene):
    digit = 9
