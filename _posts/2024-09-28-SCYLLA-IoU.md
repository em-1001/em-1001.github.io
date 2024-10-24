---
title:  "SCYLLA IoU"
excerpt: "SIoU BBox Loss"

categories:
  - Computer Vision
tags:
  - Object Detection
  - Paper
last_modified_at: 2024-09-28T08:06:00-05:00
---

# **SCYLLA-IoU(SIoU)**

SCYLLA-IoU (SIoU) considers **Angle cost**, **Distance cost**, **Shape cost** and the penalty term is as follows.

$$
\mathcal{R}_{SIoU} = \frac{\Delta + \Omega}{2}
$$

**Angle cost**

Angle cost is calculated as follows

$$
\begin{aligned}\Lambda &= 1 - 2 \cdot \sin^2\left(\arcsin(x) - \frac{\pi}{4} \right) \\   &= 1 - 2 \cdot \sin^2\left(\arcsin(\sin(\alpha)) - \frac{\pi}{4} \right) \\&= 1 - 2 \cdot \sin^2\left(\alpha - \frac{\pi}{4} \right) \\&= \cos^2\left(\alpha - \frac{\pi}{4}\right) - \sin^2\left(\alpha - \frac{\pi}{4}\right) \\ &= \cos\left(2\alpha - \frac{\pi}{2}\right) \\ &= \sin(2\alpha) \\ \end{aligned}
$$

$$
\begin{aligned} &where \\   &x = \frac{c_h}{\sigma} = \sin(\alpha) \\  &\sigma = \sqrt{(b_{c_x}^{gt} - b_{c_x})^2 + (b_{c_y}^{gt} - b_{c_y})^2} \\  &c_h = \max(b_{c_y}^{gt}, b_{c_y}) - \min(b_{c_y}^{gt}, b_{c_y})\end{aligned}
$$

If  $\alpha > \frac{\pi}{4}$ , then $\beta = \frac{\pi}{2} - \alpha$, which is calculated as beta.

**Distance cost**

Distance cost includes Angle cost, which is calculated as follows

$$
\begin{aligned}&\Delta = \sum_{t=x,y} (1 - e^{-\gamma \rho_t}) \\  &where \\  &\rho_ x = \left(\frac{b_{c_x}^{gt} - b_{c_x}}{c_w} \right)^2, \ \rho_ y = \left(\frac{b_{c_y}^{gt} - b_{c_y}}{c_h} \right)^2, \ \gamma = 2 - \Lambda\end{aligned}
$$

Here, $c_w, c_h$ are the width and height of the smallest box containing $B$ and $B^{gt}$, unlike the Angle cost.

If we look at the Distance cost, we can see that it gets sharply smaller as $\alpha \to 0$ and larger as $\alpha \to \frac{\pi}{4}$, so $\gamma$ is there to adjust it.

**Shape cost**

Shape cost is calculated as follows

$$
\begin{aligned}&\Omega = \sum_{t=w,h} (1-e^{-\omega_t})^{\theta} \\ &\\ &where \\ &\\  &\omega_w = \frac{|w-w^{gt}|}{\max(w,w^{gt})}, \omega_h = \frac{|h-h^{gt}|}{\max(h,h^{gt})} \\   \end{aligned}
$$

The $\theta$ specifies how much weight to give to the Shape cost, usually set to 4 and can be a value between 2 and 6.

The final loss is

$$
L_{SIoU} = 1 - IoU + \frac{\Delta + \Omega}{2}
$$

## SCYLLA-IoU Implementation

```py
def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint", iou_mode = "IoU", eps = 1e-7):

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    # width, height of predict and ground truth box
    w1, h1 = box1_x2 - box1_x1, box1_y2 - box1_y1 + eps
    w2, h2 = box2_x2 - box2_x1, box2_y2 - box2_y1 + eps

    # coordinates for intersection
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # width, height of convex(smallest enclosing box)
    C_w = torch.max(box1_x2, box2_x2) - torch.min(box1_x1, box2_x1)
    C_h = torch.max(box1_y2, box2_y2) - torch.min(box1_y1, box2_y1)

    # convex diagonal squared
    c2 = C_w**2 + C_h**2 + eps

    # x,y distance between the center points of the two boxes
    C_x  = torch.abs(box2_x1 + box2_x2 - box1_x1 - box1_x2) * 0.5
    C_y  = torch.abs(box2_y1 + box2_y2 - box1_y1 - box1_y2) * 0.5

    # center distance squared
    p2 = ((C_x)**2 + (C_y)**2)

    # iou
    box1_area = abs(w1 * h1)
    box2_area = abs(w2 * h2)
    union = box1_area + box2_area - intersection + eps

    iou = intersection / union

    ...

    elif iou_mode == "SIoU":
        sigma = torch.pow(p2, 0.5) + eps
        sin_alpha = C_y / sigma 
        sin_beta = C_x / sigma 
        sin_alpha = torch.where(sin_alpha > pow(2, 0.5) / 2, sin_beta, sin_alpha)
        Lambda = torch.sin(2 * torch.arcsin(sin_alpha))
        gamma = 2 - Lambda
        rho_x = (C_x / (C_w + eps))**2
        rho_y = (C_y / (C_h + eps))**2
        Delta = 2 - torch.exp(-1 * gamma * rho_x) - torch.exp(-1 * gamma * rho_y)
        omega_w = torch.abs(w1 - w2) / torch.max(w1, w2)
        omega_h = torch.abs(h1 - h2) / torch.max(h1, h2)
        Omega = torch.pow(1 - torch.exp(-1 * omega_w), 4) + torch.pow(1 - torch.exp(-1 * omega_h), 4)
        R_siou = (Delta + Omega) * 0.5

        return iou - R_siou    
```

# Reference
SIoU Loss: https://arxiv.org/pdf/2205.12740
