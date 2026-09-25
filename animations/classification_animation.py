from manim import *


import sys
from functools import lru_cache
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

NUM_PIXEL_NODES = 8  # 4 above + 4 below the ellipsis
NUM_FILTER_NODES = 6
NUM_NEURON_NODES = 7
IMAGES_PER_DIGIT = 200


# We run real MNIST test digits through the trained model and 
# record what happens inside it, using PyTorch forward hooks.

# filters = channels of the last conv block (64 channels, 5x5 each)
# neurons = the 256 hidden units in the classifier

# Every scene shows the units that contribute most to that digit,
# averaged over IMAGES_PER_DIGIT test images of it.


@lru_cache(maxsize=1)
def record_activations():
    from training.dataset import test_dataset
    from training.inference import load_model

    model = load_model(
        str(ROOT / "weights" / "model.safetensors"),
        str(ROOT / "config.json"),
    )

    activations = {}

    def save(name):
        def hook(module, inputs, output):
            activations[name] = output.detach()

        return hook

    model.features[-1].register_forward_hook(save("filters"))
    model.classifier[1].register_forward_hook(save("neurons"))

    per_digit = []

    for digit in range(10):
        indices = (test_dataset.targets == digit).nonzero().flatten()
        images = torch.stack(
            [test_dataset[i][0] for i in indices[:IMAGES_PER_DIGIT]]
        )

        with torch.no_grad():
            probabilities = torch.softmax(model(images), dim=1).mean(dim=0)

        per_digit.append(
            {
                "image": images.mean(dim=0)[0],  # (28, 28) average digit
                "filters": activations["filters"].mean(dim=0),  # (C, 5, 5)
                "neurons": activations["neurons"].mean(dim=0),  # (256,)
                "probabilities": probabilities,  # (10,)
            }
        )

    return (
        per_digit,
        model.classifier[0].weight.detach(),  # filters -> neurons
        model.classifier[2].weight.detach(),  # neurons -> outputs
    )


def pick_display(scores, num_nodes, num_active):
    """Show the num_active highest scoring units (highlighted) plus the lowest
    scoring ones as inactive context, in unit-index order."""
    order = scores.argsort(descending=True).tolist()
    active_units = order[:num_active]
    shown_units = sorted(active_units + order[-(num_nodes - num_active):])
    active_positions = [shown_units.index(unit) for unit in active_units]
    return shown_units, active_positions


@lru_cache(maxsize=10)
def digit_plan(digit):
    per_digit, w_hidden, w_out = record_activations()
    data = per_digit[digit]
    filters = data["filters"]
    num_channels = filters.shape[0]

    # How much each filter -> neuron pair pushes the neuron up:
    # sum over the 5x5 map of (filter activation x weight).
    pair = torch.einsum(
        "cs,ncs->cn",
        filters.flatten(1),
        w_hidden.view(w_hidden.shape[0], num_channels, -1),
    )  # (C, 256)

    # Neurons: activation x weight into this digit's output.
    neuron_votes = data["neurons"] * w_out[digit]
    neurons, active_neurons = pick_display(neuron_votes, NUM_NEURON_NODES, 4)
    top_neurons = [neurons[p] for p in active_neurons]

    # Filters: how much they feed the active neurons.
    filter_votes = pair[:, top_neurons].sum(dim=1)
    filters_shown, active_filters = pick_display(filter_votes, NUM_FILTER_NODES, 4)

    # Filter -> neuron links: the 2 strongest positive inputs of each active neuron.
    filter_links = []
    for n_pos in active_neurons:
        strengths = pair[[filters_shown[f] for f in active_filters], neurons[n_pos]]
        for rank in strengths.argsort(descending=True)[:2].tolist():
            if strengths[rank] > 0:
                filter_links.append((active_filters[rank], n_pos))

    # Pixel -> filter links: the 8 pixel nodes stand for 8 horizontal bands of
    # the image. Link each active filter to the inked bands where it fires most.
    bands = torch.tensor_split(data["image"], NUM_PIXEL_NODES, dim=0)
    ink = torch.stack([band.mean() for band in bands])
    pixel_links = []
    for f_pos in active_filters:
        rows = filters[filters_shown[f_pos]].mean(dim=1)  # 5 rows of the 5x5 map
        band_rows = rows[[b * len(rows) // NUM_PIXEL_NODES for b in range(NUM_PIXEL_NODES)]]
        score = band_rows * (ink > ink.mean())
        for band in score.argsort(descending=True)[:2].tolist():
            if score[band] > 0:
                pixel_links.append((band, f_pos))

    # Runner-up digits the model also considered (orange lines).
    probabilities = data["probabilities"].clone()
    probabilities[digit] = -1
    runners_up = probabilities.argsort(descending=True)[:2].tolist()
    competitor_links = []
    for other in runners_up:
        votes = torch.stack([data["neurons"][n] * w_out[other][n] for n in neurons])
        competitor_links.append((int(votes.argmax()), other))

    return {
        "active_filters": active_filters,
        "active_neurons": active_neurons,
        "pixel_links": pixel_links,
        "filter_links": filter_links,
        "competitor_links": competitor_links,
    }


class DigitClassificationScene:
    digit = 0

    def construct(self):
        self.plan = digit_plan(self.digit)

        # LAYERS
        #
        # Symmetric around x = 0:
        # -3.45 (pixels), -1.15 (strokes), +1.15 (features), +3.45 (outputs)

        input_visual, input_nodes = self.make_long_layer(
            color=WHITE,
            visible_each_side=4,
        )
        input_visual.move_to(LEFT * 4)

        strokes_layer = self.make_layer(
            NUM_FILTER_NODES,
            BLUE,
        ).move_to(LEFT * 1.15)

        hidden_layer = self.make_layer(
            NUM_NEURON_NODES,
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

        # LABELS

        input_label = Text(
            "pixels",
            font_size=36,
        ).next_to(
            input_visual,
            UP,
            buff=0.32,
        )

        strokes_label = Text(
            "filters",
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

        # CAMERA

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

        # CONNECTIONS

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

        # PREDICTION

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

        # ACTIVE NODES

        active_inputs = self.plan["active_filters"]
        active_hidden = self.plan["active_neurons"]

        # ANIMATION

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

        # PIXELS -> DETECTED STROKES

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

        # STROKES -> FEATURES

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

        # FEATURES -> OUTPUT

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

        # FINAL PREDICTION

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

    # STANDARD LAYER

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

    # HIGHLIGHTED PIXEL -> STROKE CONNECTIONS

    def make_highlighted_pixel_connections(
        self,
        pixel_nodes,
        strokes_layer,
    ):
        lines = VGroup()

        for pixel_index, stroke_index in self.plan["pixel_links"]:
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

    # ABBREVIATED 784-PIXEL LAYER

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

    # OUTPUT LAYER

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

    # BASE CONNECTIONS

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

    # HIGHLIGHTED CONNECTIONS

    def make_highlighted_connections(
        self,
        input_layer,
        hidden_layer,
        output_layer,
    ):
        feature_lines = VGroup()
        output_lines = VGroup()

        # Filters -> neurons (strongest real contributions)

        for source, target in self.plan["filter_links"]:
            feature_lines.add(
                Line(
                    input_layer[source].get_right(),
                    hidden_layer[target].get_left(),
                    color=YELLOW,
                    stroke_width=4.2,
                    stroke_opacity=0.95,
                )
            )

        # Neurons -> predicted output

        for source in self.plan["active_neurons"]:
            output_lines.add(
                Line(
                    hidden_layer[source].get_right(),
                    output_layer[self.digit].get_left(),
                    color=GREEN,
                    stroke_width=4.5,
                    stroke_opacity=0.95,
                )
            )

        # Weak competing outputs (the model's real runner-up digits)

        for source, other in self.plan["competitor_links"]:
            output_lines.add(
                Line(
                    hidden_layer[source].get_right(),
                    output_layer[other].get_left(),
                    color=ORANGE,
                    stroke_width=2.3,
                    stroke_opacity=0.55,
                )
            )

        return feature_lines, output_lines


# SCENES


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
