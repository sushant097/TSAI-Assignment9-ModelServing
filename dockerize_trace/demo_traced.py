import torch
import gradio as gr
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms


cifar10_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


@torch.jit.export
def forward_jit(model, x):
    with torch.no_grad():
        # forward pass
        logits = model(torch.Tensor(x))
        preds = F.softmax(logits, dim=-1)

    return preds


def demo():
    """Demo function.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    model = torch.jit.load("D:\EMLO_V2\Assignment\TSAI-Assignment4-Deployment-for-Demos\dockerize_trace\model.trace.pt")

    def recognize_image(image):
        if image is None:
            return None
        # transform inside the torch traced model
        image = transforms.ToTensor()(image).unsqueeze(0)
        preds = forward_jit(model, image)
        preds = preds[0].tolist()
        # print({cifar10_labels[i]: preds[i] for i in range(10)})
        return {cifar10_labels[i]: preds[i] for i in range(10)}

    im = gr.Image(shape=(32, 32), type="pil")

    demo = gr.Interface(
        fn=recognize_image,
        inputs=im,
        outputs=[gr.Label(num_top_classes=10)],
    )

    demo.launch(server_port=8080, share=True)


if __name__ == "__main__":
    demo()