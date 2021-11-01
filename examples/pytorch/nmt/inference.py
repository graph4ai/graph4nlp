import os
import resource
import nltk
import torch

from graph4nlp.pytorch.data.dataset import Text2TextDataItem
from graph4nlp.pytorch.inference_wrapper.generator_inference_wrapper import (
    GeneratorInferenceWrapper,
)
from graph4nlp.pytorch.models.graph2seq import Graph2Seq

from args import get_args
from dataset import IWSLT14Dataset

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)


def remove_bpe(str_with_subword):
    if isinstance(str_with_subword, list):
        return [remove_bpe(ss) for ss in str_with_subword]
    symbol = "@@ "
    return str_with_subword.replace(symbol, "").strip()


if __name__ == "__main__":
    opt = get_args()
    if opt["use_gpu"] != 0 and torch.cuda.is_available():
        print("[ Using CUDA ]")
        device = torch.device("cuda" if opt["gpu"] < 0 else "cuda:%d" % opt["gpu"])
    else:
        print("[ Using CPU ]")
        device = torch.device("cpu")
    model = Graph2Seq.load_checkpoint(
        os.path.join("examples/pytorch/nmt/save", opt["name"]), "best.pth"
    ).to(device)

    wrapper = GeneratorInferenceWrapper(
        cfg=opt,
        model=model,
        dataset=IWSLT14Dataset,
        data_item=Text2TextDataItem,
        beam_size=3,
        lower_case=True,
        tokenizer=nltk.RegexpTokenizer(" ", gaps=True).tokenize,
    )

    output = wrapper.predict(
        raw_contents=[
            "wissen sie , eines der großen vern@@ ü@@ gen beim reisen und eine der freu@@ den \
                bei der eth@@ no@@ graph@@ ischen forschung ist , gemeinsam mit den menschen \
                zu leben , die sich noch an die alten tage erinnern können . die ihre \
                vergangenheit noch immer im wind spüren , sie auf vom regen ge@@ gl@@ ä\
                @@ t@@ teten st@@ einen berü@@ hren , sie in den bit@@ teren blä@@ ttern \
                der pflanzen schme@@ cken ."
        ],
        batch_size=1,
    )
    output = remove_bpe(output)
    print(output)
