import torch

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def box_cxcywh_to_xyxy(x):
    x, y, w, h = x.unbind(-1)
    b = [(x - (w / 2)), (y - (h / 2)), (x + (w / 2)), (y + (h / 2))]
    return torch.stack(b, dim=-1)

def boxes_to_pixel_coords(boxes, W, H):
    boxes_x = (boxes[:,::2] * W).ceil().int()
    boxes_y = (boxes[:,1::2] * H).ceil().int()
    boxes = torch.stack([boxes_x[:,0], boxes_y[:,0], boxes_x[:,1], boxes_y[:,1]], dim=1)
    return boxes

def clamp_coords(c0, c1, max_size, min_size=0):
    if c0 < 0:
        c1 -= c0
        c0 = torch.tensor(0)
    elif c1 > max_size:
        c0 -= c1 - max_size
        c1 = torch.tensor(max_size)
    return c0, c1

def box_to_square(box, max_size, padding=2, min_size=0):
    x0, y0, x1, y1 = box
    w = x1 - x0
    h = y1 - y0
    if w > h:
        internal_padding = (w - h) / 2
        y0 -= internal_padding.floor().int() + padding
        y1 += internal_padding.ceil().int() + padding
        y0, y1 = clamp_coords(y0, y1, max_size=max_size)
        x0 -= padding
        x1 += padding
        x0, x1 = clamp_coords(x0, x1, max_size=max_size)
    elif h > w:
        internal_padding = (h - w) / 2
        x0 -= internal_padding.floor().int() + padding
        x1 += internal_padding.ceil().int() + padding
        x0, x1 = clamp_coords(x0, x1, max_size=max_size)
        y0 -= padding
        y1 += padding
        y0, y1 = clamp_coords(y0, y1, max_size=max_size)
    return torch.stack([x0, y0, x1, y1])