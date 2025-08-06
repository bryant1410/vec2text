#!/usr/bin/env python
import torch

import vec2text

CHECKPOINT_PATH = "saves/gtr-4/checkpoint-1709700"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    inversion_model = vec2text.models.InversionModel.from_pretrained(CHECKPOINT_PATH).to(DEVICE)
    inversion_model.eval()

    # inversion_trainer = vec2text.trainers.InversionTrainer(
    #     model=inversion_model, train_dataset=None, eval_dataset=None,
    #     data_collator=transformers.DataCollatorForSeq2Seq(inversion_model.tokenizer, label_pad_token_id=-100),
    # )

    inputs = inversion_model.embedder_tokenizer(
        ["A cat sat on the mat.", "A dog sat on the mat.", "This is not correct."],
        return_tensors="pt", max_length=inversion_model.config.max_seq_length, truncation=True, padding="max_length",
    )
    inputs = inputs.to(DEVICE)
    with torch.no_grad():
        frozen_embeddings = inversion_model.call_embedding_model(
            input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
        )

    gen_kwargs = {"early_stopping": False, "num_beams": 1, "do_sample": False, "no_repeat_ngram_size": 1,
                  "min_length": 1, "max_length": inversion_model.config.max_seq_length}

    regenerated = inversion_model.generate(inputs={"frozen_embeddings": frozen_embeddings},
                                           generation_kwargs=gen_kwargs)
    output_strings = inversion_model.tokenizer.batch_decode(regenerated, skip_special_tokens=True)
    print(output_strings)


if __name__ == "__main__":
    main()
