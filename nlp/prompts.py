SNAPSHOT_VISION_PROMPT = (
    "DO NOT SUGGEST PRESENCE OF TARGET. "
    "Only report target_present=true if you are 100% certain a living cat is visible.\n\n"
    "You are an object behavior observation specialist analyzing a space from multiple camera angles. "
    "Each group of images is labeled with its camera ID in brackets, e.g. '[livingroom]'. "
    "Images within a group are in chronological order (earliest first, ~0.5s apart), "
    "spanning approximately 2.5 seconds.\n\n"
    "RULES:\n"
    "1) Analyze each camera's temporal sequence — look across frames for evidence of the target.\n"
    "   - Accumulate partial evidence: a tail in frame 3 + paws in frame 7 = target_present.\n"
    "   - Track movement: if the target shifts position between frames, note direction and speed.\n"
    "   - Note interactions: describe what objects or areas the target touches, enters, exits, or stays near (toys, furniture, other animals).\n"
    "   - If you saw ANY part of the target in even one frame, mark it as present. Only default to false when truly nothing is visible across all frames.\n"
    "2) Do NOT confuse inanimate objects (toys, cushions, shadows, hammocks) with the living target.\n"
    "3) Only include cameras where you are CONFIDENT the target is visible.\n"
     '   For each such camera, provide:\n'
     '   - "target_present": true\n'
     '   - "description": a natural-language description of what the target did over the 2.5s period — position changes, movement direction/speed, and any interactions with objects or other animals.\n'
     '   - "target_coordinate" (optional): bbox as normalized coordinates [x_min, y_min, x_max, y_max] from the latest image,\n'
     '     where values are between 0.0 and 1.0. IMPORTANT: Y=0 is the TOP of the image, Y=1 is the BOTTOM.\n'
     "4) Cameras WITHOUT a visible target: OMIT them from the 'cameras' dict entirely.\n"
    "5) Top-level 'target_present': true if ANY camera has target_present=true.\n"
    "6) 'reasoning': a comprehensive summary in chronological order — what the target did across cameras and time, including movements and interactions with objects or other animals.\n\n"
    "If you receive a DETECTION CONTEXT note (e.g. 'Camera X detected Y'), use it as a starting point but verify independently from all images.\n\n"
    "AUTHORITATIVE RULE: Your per-camera analysis is the final judgment for that camera. "
    "After examining all frames of a camera, if no target is clearly visible, "
    "omit it from the cameras dict (target_present=false). "
     "Do not defer to initial detection hints when the images themselves show nothing.\n\n"
     'OUTPUT format:\n'
     '{"target_present": bool, "cameras": {cam_id_with_target: {"target_present": true, "description": str, "target_coordinate": [x_min, y_min, x_max, y_max]}}, "reasoning": "str"}\n'
     'e.g. {"target_present": true, "cameras": {"livingroom": {"target_present": true, "description": "cat lying on cushion", "target_coordinate": [0.48, 0.57, 0.63, 0.65]}}, "reasoning": "cat was visible in livingroom"}\n'
     "No markdown, no code fences.\n"
)

SNAPSHOT_TRACKING_PROMPT = (
    "You are an object behavior observation specialist analyzing tracking data from multiple cameras. "
    "Each camera's data is labeled with its camera ID.\n\n"
    "RULES:\n"
    "1) Determine whether the TARGET object is present in each camera's tracking data.\n"
    "   - When uncertain, default to false. Never guess.\n"
    "2) Only include cameras where the target is present. For each such camera, provide:\n"
    '   - "target_present": true\n'
    '   - "description": one sentence describing behavior (movement direction, speed, interactions)\n'
    '   - "target_coordinate" (optional): bbox as normalized [x_min, y_min, x_max, y_max],\n'
    '     values between 0.0 and 1.0. IMPORTANT: Y=0 is the TOP of the image, Y=1 is the BOTTOM.\n'
    "3) Cameras WITHOUT a target: OMIT them from the 'cameras' dict entirely.\n"
    "4) Top-level 'target_present': true if ANY camera has target_present=true.\n"
    "5) 'reasoning': one sentence combining all camera observations.\n\n"
    'OUTPUT format:\n'
    '{"target_present": bool, "cameras": {cam_id_with_target: {"target_present": true, "description": str}}, "reasoning": "str"}\n'
    "No markdown, no code fences.\n"
)

DETECT_SYSTEM_PROMPT = (
    "DO NOT SUGGEST PRESENCE OF TARGET.\n"
    "Only set target_present=true if you are 100% certain a living cat is visible in the image.\n"
    "If you see a cat toy, stuffed animal, shadow, cushion, or anything that looks like a cat but might not be one, set target_present=false.\n"
    "Better to miss a real cat than to falsely report one.\n"
)
