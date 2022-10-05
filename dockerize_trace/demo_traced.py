import torch
import gradio as gr
import numpy as np



cifar10_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def demo():
    """Demo function.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    model = torch.jit.load("model_script.pt")

    def recognize_image(image):
        if image is None:
            return None
        # transform inside the torch traced model
        image = np.expand_dim(image, axis=0)
        preds = model.forward_jit(image)
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