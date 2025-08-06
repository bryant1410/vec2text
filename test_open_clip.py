#!/usr/bin/env python
import transformers

import vec2text


def main() -> None:
    last_checkpoint = transformers.trainer_utils.get_last_checkpoint(
        "saves/openclip_vit_b_32_quickgelu_openai_1"
    )
    print(last_checkpoint)

    inversion_model = vec2text.models.InversionModel.from_pretrained(last_checkpoint)

    # This is incorrect, but it doesn't matter for the test:
    corrector_model = vec2text.models.CorrectorEncoderModel.from_pretrained(
        last_checkpoint
    )

    corrector = vec2text.load_corrector(inversion_model, corrector_model)

    print(
        vec2text.invert_strings(
            [
                "A photo of a cat",
                "A dog jumping over a fence",
                "A video frame of an octopus playing the bass.",
                "A photo of a steak",
                "choripan",
            ],
            corrector=corrector,
        )
    )


if __name__ == "__main__":
    main()
