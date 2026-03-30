# --------------------------------------------------------
# Based on SeeGround (Apache 2.0)
# Query ablation with target/relation/caption variants
# Computes CLIP similarity between queries and multi-view images and stores the results
# --------------------------------------------------------

import sys
import argparse
import os
import random
import numpy as np
import open3d as o3d
from tqdm import tqdm
from openai import OpenAI
import json
import csv
import torch, gc
from pytorch3d.structures import Pointclouds

import clip
from PIL import Image
import torch.nn.functional as F


sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from inference.multiview_projection import ( # 수정
    render_point_cloud_with_pytorch3d_with_objects_multiview,
    setup_camera,
    create_point_cloud,
    )

from inference.utils import (
    parse_response,
    calc_iou,
    encode_img,
    read_file_to_list,
    save_to_file,
    stem_match,
    fuzzy_match,
    load_json,
    load_bboxes,
    generate_objects_info,
    load_scene_pcd,
)

SYSTEM_INFO = "You are a helpful assistant designed to identify objects based on image and descriptions."
COOR_INFO = "The 3D spatial coordinate system is defined as follows: X-axis and Y-axis represent horizontal dimensions, Z-axis represents the vertical dimension."
ASK_INFO = "Please review the provided images and object 3D spatial descriptions, then select the object ID that best matches the given description. there are multi-view images and you should consider spatial informations."
RESPONSE_FORMAT = "Respond in the format: 'Predicted ID: <ID>\nExplanation: <explanation>', where <ID> is the object ID and <explanation> is your reasoning."

def create_openai_messages(query, objects_info, use_image=False, image_path=None):
    """Create OpenAI API messages."""
    messages = [
        {"role": "system", "content": SYSTEM_INFO},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{COOR_INFO}\n\nObject IDs and their positions:\n{objects_info}\n\n{ASK_INFO}\n\n{RESPONSE_FORMAT}\n\nThe given description is: {query}",
                }
            ],
        },
    ]

    if use_image and image_path:
        img_url = encode_img(image_path)
        messages[1]["content"][0] = {
            "type": "text",
            "text": f"As shown in the image, this is a rendered image of a room, and the picture reflects your current view. Each object in the room is labeled by a unique number (ID) in red color on its surface. \n\nObject IDs and their 3D spatial information are as follows:\n{objects_info}\n\n{COOR_INFO}\n\n{ASK_INFO}\n\n{RESPONSE_FORMAT}\n\nThe given description is: {query}",
        }
        messages[1]["content"].insert(
            0, {"type": "image_url", "image_url": {"url": img_url}}
        )

    return messages

def encode_queries_with_clip(clip_model, queries, device):
    """
    queries: list of strings
    return: list of normalized CLIP text features
    """
    text_features_list = []

    for q in queries:
        q = q.strip()
        if len(q) == 0:
            continue

        # tokenize
        tokens = clip.tokenize([q]).to(device)

        tokens = tokens[:, :77]

        with torch.no_grad():
            f = clip_model.encode_text(tokens)
            f = f / f.norm(dim=-1, keepdim=True)

        text_features_list.append(f)

    return text_features_list

def process_query(
    query,
    objects_info,
    openai_api_key,
    openai_api_base,
    use_image=False,
    image_path=None,  
    model_name="Qwen2-VL-7B-Instruct", 
    log_file=None,
):
    """Process query and return model's response."""
    assert objects_info is not None
    assert query is not None

    client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

    messages = create_openai_messages(query, objects_info, use_image, image_path)

    # Save the input messages to a file
    if log_file and not os.path.exists(log_file):
        save_to_file(log_file, str(messages))

    chat_response = client.chat.completions.create(model=model_name, messages=messages)
    result = chat_response.choices[0].message.content
    return result.replace("\\n", "\n")

def process_room(
    dataset,
    room,
    pcd_dir,
    split,
    output_dir,
    language_annotation_file,
    gt_bbox_dir,
    pred_bbox_dir,
    openai_api_key,
    openai_api_base,
    use_image=False,
    model_name=None,
    verbose=True,
):
    """Process a single room with queries and predictions."""
    # Load annotations and bounding boxes
    data = load_json(language_annotation_file)
    queries = [it for it in data if it["scan_id"] == room]
    gt_bboxes = load_bboxes(room, gt_bbox_dir, "gt")
    mask3d_bboxes = load_bboxes(room, pred_bbox_dir, "pred")
    object_names = [obj["target"] for obj in mask3d_bboxes.values()]

    # Generate objects information
    objects_info = generate_objects_info(mask3d_bboxes.values())

    output_file = os.path.join(output_dir, "pred", f"{room}.json")
    if os.path.exists(output_file):
        print(f"File {output_file} already exists, skipping")
        return
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    log_file = os.path.join(output_dir, "room_info", f"{room}.txt")
    print(f"Saved objects_info to {log_file}")
    

    correct_predictions = 0
    total_predictions = 0  # len(queries)
    results = []

    queries = sorted(queries, key=lambda x: int(x["target_id"]))

    for i, d in enumerate(tqdm(queries)):
        torch.cuda.empty_cache()
        gc.collect()
        
        query = d["caption"]
        gt_id = int(d["target_id"])
        image_path = None
        print()
        print(f"Query: {query}")
        
        # Matching target and anchor
        try:
            target_name = d["parsed_query"]["Target"]
            anchor_name = d["parsed_query"]["Anchor"]
        except:
            target_name = ""
            anchor_name = ""
        print(f"Parsed target: {target_name}; anchor: {anchor_name}")
        matched_targets = fuzzy_match(target_name, object_names).union(
            stem_match(target_name, object_names)
        )
        matched_anchors = fuzzy_match(anchor_name, object_names).union(
            stem_match(anchor_name, object_names)
        )
        print(f"Matched target: {matched_targets}; anchor: {matched_anchors}")

        ## reconstructed Query
            
        # ----- 1. Safe extraction -----
        target_name = d["parsed_query"].get("Target", "")
        target_des  = d["parsed_query"].get("TargetDescription", "")
        anchor_raw  = d["parsed_query"].get("Anchor", "")
        relation_raw = d["parsed_query"].get("Relation", "")

        # Normalize to list
        anchors_name = anchor_raw if isinstance(anchor_raw, list) else [anchor_raw] if anchor_raw else []
        relations = relation_raw if isinstance(relation_raw, list) else [relation_raw] if relation_raw else []

        print("Parsed:", target_name, target_des, anchors_name, relations)

        # ----- 2. Query construction -----

        # target_des + target_name
        target_query = f"{target_des} {target_name}".strip()

        # target_name + relation + anchor
        anchor_queries = []
        for a, r in zip(anchors_name, relations):
            phrase = f"{target_name} {r} {a}".strip()
            anchor_queries.append(phrase)

        # If empty, ignore safely
        print("target_query =", target_query)
        print("anchor_queries =", anchor_queries)
        
        targets = [
            obj for obj in mask3d_bboxes.values() if obj["target"] in matched_targets
        ]
        anchors = [
            obj for obj in mask3d_bboxes.values() if obj["target"] in matched_anchors
        ]

        if len(targets) == 0:
            targets = list(mask3d_bboxes.values())

        if len(anchors) == 0:
            anchors = targets
            
        # generate multi-view images 
        scan_pc, center = load_scene_pcd(room, pcd_dir)
        image_path_list = []
        image_path_list, cameras_list = render_point_cloud_with_pytorch3d_with_objects_multiview( # list 반환 f_name = f"{save_dir}/rendered{i}.png"
            mask3d_bboxes.values(),
            targets=targets,
            anchors=anchors,
            center=center,
            scan_pc=scan_pc,
            save_dir=f"./multiview_CLIP/{dataset}/qwen2-vl-7b/{room}/{i}",

            image_size=680,
            draw_id=True,
            draw_img=True,
            # draw_mask=True,
            # draw_contour=True,
        )
        
        # Process query with OpenAI
        for j, image_path in enumerate(image_path_list): 
            # text features
            target_feature = encode_queries_with_clip(clip_model, queries=target_query, device=device)
            anchor_feature = encode_queries_with_clip(clip_model, queries=anchor_queries, device=device)
            
            # collect scores for this query
            scores_for_query = {}
            
            # image features
            img = Image.open(image_path)
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                img_feats = clip_model.encode_image(img_tensor)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            
            # target score (single)
            if len(target_feature) > 0:
                target_score = F.cosine_similarity(img_feats, target_feature[0], dim=-1).item()
            else:
                target_score = 0.0

            # anchor scores (list)
            anchor_scores = []
            for f in anchor_feature:
                s = F.cosine_similarity(img_feats, f, dim=-1).item()
                anchor_scores.append(s)

            # anchor score aggregate: avg score
            anchor_score_final = sum(anchor_scores)/len(anchor_scores) if len(anchor_scores) > 0 else 0.0

            # final score
            alpha = 0.7 
            final_score = alpha * target_score + (1-alpha) * anchor_score_final

            scores_for_query[image_path] = final_score

            print(f"Image: {image_path}")
            print(f"  Target score:  {target_score:.4f}")
            print(f"  Anchor score:  {anchor_score_final:.4f}")
            print(f"  Final score:   {final_score:.4f}")
            
            torch.cuda.empty_cache()
            
            total_predictions += 1

            # Store results
            results.append(
                {
                    "query_id": i,
                    "image_id": j, 
                    "query": query,
                    "image_path": image_path,
                    "parsed_query": d["parsed_query"],
                    "target_score": target_score,
                    "anchor_score_final": anchor_score_final,
                    "final_score": final_score,
                    "room": room,
                }
            )
    
    save_to_file(output_file, json.dumps(results, indent=4))
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scanrefer", help="Dataset name")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument(
        "--output_dir",
        # required=True,
        default="./outputs/qwen2-vl-7b/scanrefer/multi-view_query/pred",
        help="Directory to store the output",
    )
    parser.add_argument(
        "--language_annotation_dir",
        # required=True,
        default="./data/scanrefer/query_rel_7B",
        help="Parsed language annotation (with anchor and target) file path",
    )
    parser.add_argument(
        "--gt_bbox_dir",
        # required=True,
        default="./data/scanrefer/object_lookup_table/gt",
        help="Ground truth bounding box directory",
    )
    parser.add_argument(
        "--pred_bbox_dir",
        # required=True,
        default="./data/scanrefer/object_lookup_table/pred",
        help="Predicted bounding box directory",
    )
    parser.add_argument(
        "--pcd_dir",
        # required=True,
        default='./data/referit3d/scan_data/pcd_with_global_alignment',
        help="",
    )
    parser.add_argument(
        "--openai_api_key", 
        default="OPENAI-API-KEY", 
        help="OpenAI API Key"
    )
    parser.add_argument(
        "--openai_api_base",
        default="http://localhost:8000/v1",
        help="OpenAI API Base URL",
    )
    parser.add_argument(
        "--use_image",
        default=True,
        help="Whether to use image rendering",
    )
    parser.add_argument(
        "--model_name", default="Qwen2-VL-7B-Instruct", help="Model name"
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="./data/scannet/scannetv2_val.txt",
        help="Path to the validation split file.",
    )

    args = parser.parse_args()

    scan_ids = list(os.listdir(args.language_annotation_dir))
    scan_ids = sorted([i.split(".")[0] for i in scan_ids])
    print(f"Found {len(scan_ids)} scans in {args.language_annotation_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # clip_model, preprocess = clip.load("ViT-B/32", device=device) 
    clip_model, preprocess = clip.load("ViT-L/14@336px", device=device)  
    for room in tqdm(scan_ids, desc="Process rooms"):

        language_annotation_file = args.language_annotation_dir + f"/{room}.json"

        process_room(
            dataset=args.dataset,
            room=room,
            split=args.split,
            output_dir=args.output_dir,
            pcd_dir=args.pcd_dir,
            language_annotation_file=language_annotation_file,
            gt_bbox_dir=args.gt_bbox_dir,
            pred_bbox_dir=args.pred_bbox_dir,
            openai_api_key=args.openai_api_key,
            openai_api_base=args.openai_api_base,
            use_image=args.use_image,
            model_name=args.model_name,
        )
