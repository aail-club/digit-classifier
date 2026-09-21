"""Conceptual MNIST classification animations, one scene per predicted digit.

Render/copy examples:
    manim -ql --format=gif animations/classification_animation.py ClassificationAnimation0
    cp media/videos/classification_animation/480p15/ClassificationAnimation0_ManimCE_*.gif \
        assets/classification_animation_0.gif

Repeat for ClassificationAnimation1 ... ClassificationAnimation9 so the app can load:
    assets/classification_animation_0.gif ... assets/classification_animation_9.gif

The highlighted activations and weighted connections are illustrative only; they are not
real model weights.
"""

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
    """Mixin with the shared layout for each digit-specific scene."""

    digit = 0

    def construct(self):
        digit_group = self.make_digit_card(self.digit).to_edge(LEFT, buff=0.65)
        input_layer = self.make_layer(6, BLUE).shift(LEFT * 1.55)
        hidden_layer = self.make_layer(7, TEAL).shift(RIGHT * 0.35)
        output_nodes, output_labels = self.make_output_layer().shift(RIGHT * 2.85)

        all_connections = VGroup(
            self.connect_layers(input_layer, hidden_layer),
            self.connect_layers(hidden_layer, output_nodes),
        )
        feature_connections, output_connections = self.make_highlighted_connections(
            input_layer,
            hidden_layer,
            output_nodes,
        )

        labels = VGroup(
            Text("camera", font_size=22).next_to(digit_group, DOWN, buff=0.18),
            Text("strokes", font_size=22).next_to(input_layer, DOWN, buff=0.18),
            Text("features", font_size=22).next_to(hidden_layer, DOWN, buff=0.18),
            Text("outputs", font_size=22).next_to(output_nodes, DOWN, buff=0.18),
        )
        prediction = Text(
            f"Prediction: {self.digit}",
            font_size=40,
            weight=BOLD,
            color=GREEN,
        ).to_edge(DOWN, buff=0.32)
        arrow = Arrow(digit_group.get_right(), input_layer.get_left(), buff=0.22, color=YELLOW)

        active_inputs = DIGIT_FEATURES[self.digit]
        active_hidden = sorted({(feature * 2 + self.digit) % len(hidden_layer) for feature in active_inputs})
        active_hidden += [idx for idx in range(len(hidden_layer)) if idx not in active_hidden][:2]
        active_hidden = active_hidden[:5]

        self.play(FadeIn(digit_group), Write(labels[0]))
        self.play(GrowArrow(arrow), FadeIn(input_layer), Write(labels[1]))
        self.play(Create(all_connections), FadeIn(hidden_layer), FadeIn(output_nodes), FadeIn(output_labels))
        self.play(Write(labels[2]), Write(labels[3]))
        self.play(
            LaggedStart(
                *[input_layer[index].animate.set_fill(YELLOW, opacity=0.95) for index in active_inputs],
                lag_ratio=0.08,
            ),
            Create(feature_connections),
            run_time=0.9,
        )
        self.play(
            LaggedStart(
                *[hidden_layer[index].animate.set_fill(ORANGE, opacity=0.9) for index in active_hidden],
                lag_ratio=0.08,
            ),
            Create(output_connections),
            run_time=0.9,
        )
        self.play(
            output_nodes[self.digit].animate.set_fill(GREEN, opacity=1.0).scale(1.22),
            output_labels[self.digit].animate.set_color(GREEN).scale(1.18),
            Write(prediction),
        )
        self.wait(0.8)

    def make_digit_card(self, digit):
        digit_text = Text(str(digit), font_size=118, weight=BOLD)
        box = RoundedRectangle(width=1.55, height=2.0, corner_radius=0.14, color=BLUE)
        box.move_to(digit_text)
        return VGroup(box, digit_text)

    def make_layer(self, count, color):
        nodes = VGroup(
            *[
                Circle(radius=0.115, color=color, fill_color=color, fill_opacity=0.32)
                for _ in range(count)
            ]
        )
        nodes.arrange(DOWN, buff=0.13)
        return nodes

    def make_output_layer(self):
        nodes = self.make_layer(10, GREEN)
        labels = VGroup(
            *[
                Text(str(index), font_size=18).next_to(nodes[index], RIGHT, buff=0.12)
                for index in range(10)
            ]
        )
        return VGroup(nodes, labels)

    def connect_layers(self, left_layer, right_layer):
        return VGroup(
            *[
                Line(
                    left.get_right(),
                    right.get_left(),
                    stroke_width=0.8,
                    color=GRAY,
                    stroke_opacity=0.28,
                )
                for left in left_layer
                for right in right_layer
            ]
        )

    def make_highlighted_connections(self, input_layer, hidden_layer, output_layer):
        active_inputs = DIGIT_FEATURES[self.digit]
        feature_lines = VGroup()
        output_lines = VGroup()

        for feature in active_inputs:
            targets = [
                (feature + self.digit) % len(hidden_layer),
                (feature * 2 + self.digit + 1) % len(hidden_layer),
            ]
            for target in targets:
                feature_lines.add(
                    Line(
                        input_layer[feature].get_right(),
                        hidden_layer[target].get_left(),
                        color=YELLOW,
                        stroke_width=3,
                        stroke_opacity=0.9,
                    )
                )

        hidden_targets = sorted({(feature * 2 + self.digit) % len(hidden_layer) for feature in active_inputs})[:4]
        for source in hidden_targets:
            output_lines.add(
                Line(
                    hidden_layer[source].get_right(),
                    output_layer[self.digit].get_left(),
                    color=GREEN,
                    stroke_width=3.2,
                    stroke_opacity=0.95,
                )
            )

        nearby_digits = {max(0, self.digit - 1), min(9, self.digit + 1)} - {self.digit}
        for nearby in nearby_digits:
            source = (nearby + self.digit) % len(hidden_layer)
            output_lines.add(
                Line(
                    hidden_layer[source].get_right(),
                    output_layer[nearby].get_left(),
                    color=ORANGE,
                    stroke_width=1.6,
                    stroke_opacity=0.55,
                )
            )

        return feature_lines, output_lines


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
