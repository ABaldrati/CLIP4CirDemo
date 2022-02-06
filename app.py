import json
import pickle
import random
import PIL.Image
import PIL.ImageOps
import clip
import numpy as np
import torch
from flask import Flask, send_file, url_for
from flask import render_template, request, redirect
from io import BytesIO
from typing import Optional, Tuple, Union
from data_utils import targetpad_resize, server_base_path, data_path
from model import Combiner

app = Flask(__name__)

if torch.cuda.is_available():
    device = torch.device("cuda")
    data_type = torch.float16
else:
    device = torch.device("cpu")
    data_type = torch.float32


@app.route('/')
def choice():
    """
    Makes the render of dataset_choice template.
    """
    return render_template('dataset_choice.html')


@app.route('/favicon.ico')
def favicon():
    return url_for('static', filename='/favicon.ico')


@app.route('/<string:dataset>')
def reference(dataset: str):
    """
    Get 30 random reference images and makes the render of the 'reference' template
    :param dataset: dataset where get the reference images
    """
    if dataset == 'cirr':
        random_indexes = random.sample(range(len(cirr_val_triplets)), k=30)
        triplets = np.array(cirr_val_triplets)[random_indexes]
        names = [triplet['reference'] for triplet in triplets]
    elif dataset == 'fashionIQ':
        random_indexes = random.sample(range(len(fashionIQ_val_triplets)), k=30)
        triplets = np.array(fashionIQ_val_triplets)[random_indexes]
        names = [triplet['candidate'] for triplet in triplets]
    else:
        return redirect(url_for('choice'))
    return render_template('reference.html', dataset=dataset, names=names)


@app.route('/<string:dataset>/<string:reference_name>')
def relative_caption(dataset: str, reference_name: str):
    """
    Get the dataset relative captions for the given reference image and renders the 'relative_caption' template
    :param dataset: dataset of the reference image
    :param reference_name: name of the reference images
    """
    relative_captions = []
    if dataset == 'cirr':
        for triplet in cirr_val_triplets:
            if triplet['reference'] == reference_name:
                relative_captions.append(f"{triplet['caption']}")
    elif dataset == 'fashionIQ':
        for triplet in fashionIQ_val_triplets:
            if triplet['candidate'] == reference_name:
                relative_captions.append(
                    f"{triplet['captions'][0].strip('?,. ').capitalize()} and {triplet['captions'][1].strip('?,. ')}")
    return render_template('relative_caption.html', dataset=dataset, reference_name=reference_name,
                           relative_captions=relative_captions)


@app.route('/<string:dataset>/<string:reference_name>', methods=['POST'])
def relative_caption_post(dataset: str, reference_name: str):
    """
    Get the custom caption with a POST method and makes the render of 'results' template
    :param dataset: dataset of the query
    :param reference_name: reference image name
    """
    caption = request.form['custom_caption']
    return redirect(url_for('results', dataset=dataset, reference_name=reference_name, caption=caption))


@app.route('/<string:dataset>/<string:reference_name>/<string:caption>')
def results(dataset: str, reference_name: str, caption: str):
    """
    Compute the results of a given query and makes the render of 'results.html' template
    :param dataset: dataset of the query
    :param reference_name: reference image name
    :param caption: relative caption
    """
    n_retrieved = 50  # retrieve first 50 results since for both dataset the R@50 is the broader scale metric

    if dataset == 'cirr':
        combiner = cirr_combiner
    elif dataset == 'fashionIQ':
        combiner = fashionIQ_combiner
    else:
        raise ValueError()
    sorted_group_names = ""

    if dataset == 'cirr':
        # Compute CIRR results
        sorted_group_names, sorted_index_names, target_name = compute_cirr_results(caption, combiner, n_retrieved,
                                                                                   reference_name)
    elif dataset == "fashionIQ":
        # Compute fashionIQ results
        sorted_index_names, target_name = compute_fashionIQ_results(caption, combiner, n_retrieved, reference_name)

    else:
        return redirect(url_for('choice'))

    return render_template('results.html', dataset=dataset, caption=caption, reference_name=reference_name,
                           names=sorted_index_names[:n_retrieved], target_name=target_name,
                           group_names=sorted_group_names)


def compute_fashionIQ_results(caption: str, combiner: Combiner, n_retrieved: int, reference_name: str) -> Tuple[
    np.array, str]:
    """
    Combine visual-text features and compute fashionIQ results
    :param caption: relative caption
    :param combiner: fashionIQ Combiner network
    :param n_retrieved: number of images to retrieve
    :param reference_name: reference image name
    :return: Tuple made of: 1)top 'n_retrieved' index names , 2) target_name (when known)
    """

    target_name = ""
    dress_type = ""

    # Assign the correct FashionIQ category to the reference image
    if reference_name in fashionIQ_dress_index_names:
        dress_type = 'dress'
    elif reference_name in fashionIQ_toptee_index_names:
        dress_type = 'toptee'
    elif reference_name in fashionIQ_shirt_index_names:
        dress_type = 'shirt'

    # Check if the query belongs to the validation set and get query info
    for triplet in fashionIQ_val_triplets:
        if triplet['candidate'] == reference_name:
            if f"{triplet['captions'][0].strip('?,. ').capitalize()} and {triplet['captions'][1].strip('?,. ')}" == caption:
                target_name = triplet['target']
                dress_type = triplet['dress_type']

    # Get the right category index features
    if dress_type == "dress":
        if target_name == "":
            index_names = fashionIQ_dress_index_names
            index_features = fashionIQ_dress_index_features.to(device)
        else:
            index_features = fashionIQ_val_dress_index_features.to(device)
            index_names = fashionIQ_val_dress_index_names
        reference_index = index_names.index(reference_name)
        index_features = index_features.to(device)
    elif dress_type == "toptee":
        if target_name == "":
            index_names = fashionIQ_toptee_index_names
            index_features = fashionIQ_toptee_index_features.to(device)
        else:
            index_features = fashionIQ_val_toptee_index_features.to(device)
            index_names = fashionIQ_val_toptee_index_names
        reference_index = index_names.index(reference_name)
        index_features = index_features.to(device)
    elif dress_type == "shirt":
        if target_name == "":
            index_names = fashionIQ_shirt_index_names
            index_features = fashionIQ_shirt_index_features.to(device)
        else:
            index_features = fashionIQ_val_shirt_index_features.to(device)
            index_names = fashionIQ_val_shirt_index_names
        reference_index = index_names.index(reference_name)
        index_features = index_features.to(device)
    else:
        raise ValueError()

    # Get visual features, extract textual features and compute combined features
    reference_features = index_features[reference_index].unsqueeze(0)
    text_inputs = clip.tokenize(caption, truncate=True).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_inputs)
        predicted_features = combiner.combine_features(reference_features, text_features).squeeze(0)

    # Sort the results and get the top 50
    cos_similarity = index_features @ predicted_features.T
    sorted_indices = torch.topk(cos_similarity, n_retrieved, largest=True).indices.cpu()
    sorted_index_names = np.array(index_names)[sorted_indices].flatten()

    return sorted_index_names, target_name


def compute_cirr_results(caption: str, combiner: Combiner, n_retrieved: int, reference_name: str) -> Tuple[
    Union[list, object], np.array, str]:
    """
    Combine visual-text features and compute CIRR results
    :param caption: relative caption
    :param combiner: CIRR Combiner network
    :param n_retrieved: number of images to retrieve
    :param reference_name: reference image name
    :return: Tuple made of: 1) top group index names (when known) 2)top 'n_retrieved' index names , 3) target_name (when known)
    """
    target_name = ""
    group_members = ""
    sorted_group_names = ""

    index_features = cirr_index_features.to(device)
    index_names = cirr_index_names

    # Check if the query belongs to the validation set and get query info
    for triplet in cirr_val_triplets:
        if triplet['reference'] == reference_name and triplet['caption'] == caption:
            target_name = triplet['target_hard']
            group_members = triplet['img_set']['members']
            index_features = cirr_val_index_features.to(device)
            index_names = cirr_val_index_names

    # Get visual features, extract textual features and compute combined features
    text_inputs = clip.tokenize(caption, truncate=True).to(device)
    reference_index = index_names.index(reference_name)
    reference_features = index_features[reference_index].unsqueeze(0)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_inputs)
        predicted_features = combiner.combine_features(reference_features, text_features).squeeze(0)

    # Sort the results and get the top 50
    cos_similarity = index_features @ predicted_features.T
    sorted_indices = torch.topk(cos_similarity, n_retrieved, largest=True).indices.cpu()
    sorted_index_names = np.array(index_names)[sorted_indices].flatten()
    sorted_index_names = np.delete(sorted_index_names, np.where(sorted_index_names == reference_name))

    # If it is a validation set query compute also the group results
    if group_members != "":
        group_indices = [index_names.index(name) for name in group_members]
        group_features = index_features[group_indices]
        cos_similarity = group_features @ predicted_features.T
        group_sorted_indices = torch.argsort(cos_similarity, descending=True).cpu()
        sorted_group_names = np.array(group_members)[group_sorted_indices]
        sorted_group_names = np.delete(sorted_group_names, np.where(sorted_group_names == reference_name)).tolist()

    return sorted_group_names, sorted_index_names, target_name


@app.route('/get_image/<string:image_name>')
@app.route('/get_image/<string:image_name>/<int:dim>')
@app.route('/get_image/<string:image_name>/<int:dim>/<string:gt>')
@app.route('/get_image/<string:image_name>/<string:gt>')
def get_image(image_name: str, dim: Optional[int] = None, gt: Optional[str] = None):
    """
    get CIRR or FashionIQ Image
    :param image_name: image name
    :param dim: size to resize the image
    :param gt: if 'true' the has a green border, if 'false' has a red border anf if 'none' has a tiny black border
    """

    # Check whether the image comes from CIRR or FashionIQ dataset
    if image_name in cirr_name_to_relpath:  #
        image_path = server_base_path / 'cirr_dataset' / f'{cirr_name_to_relpath[image_name]}'
    elif image_name in fashion_index_names:
        image_path = server_base_path / 'fashionIQ_dataset' / 'images' / f"{image_name}.jpg"
    else:
        raise ValueError()

    # if 'dim' is not None resize the image
    if dim:
        transform = targetpad_resize(1.25, int(dim), 255)
        pil_image = transform(PIL.Image.open(image_path))
    else:
        pil_image = PIL.Image.open(image_path)

    # add a border to the image
    if gt == 'True':
        pil_image = PIL.ImageOps.expand(pil_image, border=5, fill='green')
    elif gt == 'False':
        pil_image = PIL.ImageOps.expand(pil_image, border=5, fill='red')
    elif gt is None:
        pil_image = PIL.ImageOps.expand(pil_image, border=1, fill='grey')

    img_io = BytesIO()
    pil_image.save(img_io, 'JPEG', quality=80)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')


@app.before_first_request
def _load_assets():
    """
    Load all the necessary assets
    """

    # load CIRR assets ---------------------------------------------------------------------------------------
    load_cirr_assets()

    # load FashionIQ assets ------------------------------------------------------------------------------
    load_fashionIQ_assets()

    # Load CLIP model and Combiner networks
    global clip_model
    global clip_preprocess
    clip_model, clip_preprocess = clip.load("RN50x4")
    clip_model = clip_model.eval().to(device)

    global fashionIQ_combiner
    fashionIQ_combiner = torch.hub.load(server_base_path, source='local', model='combiner', dataset='fashionIQ')
    fashionIQ_combiner = torch.jit.script(fashionIQ_combiner).type(data_type).to(device).eval()

    global cirr_combiner
    cirr_combiner = torch.hub.load(server_base_path, source='local', model='combiner', dataset='cirr')
    cirr_combiner = torch.jit.script(cirr_combiner).type(data_type).to(device).eval()


def load_fashionIQ_assets():
    """
    Load fashionIQ assets
    """
    global fashionIQ_val_triplets
    fashionIQ_val_triplets = []
    for dress_type in ['dress', 'toptee', 'shirt']:
        with open(server_base_path / 'fashionIQ_dataset' / 'captions' / f'cap.{dress_type}.val.json') as f:
            dress_type_captions = json.load(f)
            captions = [dict(caption, dress_type=f'{dress_type}') for caption in dress_type_captions]
            fashionIQ_val_triplets.extend(captions)

    global fashionIQ_val_dress_index_features
    fashionIQ_val_dress_index_features = torch.load(
        data_path / 'fashionIQ_val_dress_index_features.pt', map_location=device).type(data_type).cpu()

    global fashionIQ_val_dress_index_names
    with open(data_path / 'fashionIQ_val_dress_index_names.pkl', 'rb') as f:
        fashionIQ_val_dress_index_names = pickle.load(f)

    global fashionIQ_test_dress_index_features
    fashionIQ_test_dress_index_features = torch.load(
        data_path / 'fashionIQ_test_dress_index_features.pt', map_location=device).type(data_type).cpu()

    global fashionIQ_test_dress_index_names
    with open(data_path / 'fashionIQ_test_dress_index_names.pkl', 'rb') as f:
        fashionIQ_test_dress_index_names = pickle.load(f)

    global fashionIQ_dress_index_names
    global fashionIQ_dress_index_features
    fashionIQ_dress_index_features = torch.vstack(
        (fashionIQ_val_dress_index_features, fashionIQ_test_dress_index_features))
    fashionIQ_dress_index_names = fashionIQ_val_dress_index_names + fashionIQ_test_dress_index_names

    global fashionIQ_val_shirt_index_features
    fashionIQ_val_shirt_index_features = torch.load(
        data_path / 'fashionIQ_val_shirt_index_features.pt', map_location=device).type(data_type).cpu()

    global fashionIQ_val_shirt_index_names
    with open(data_path / 'fashionIQ_val_shirt_index_names.pkl', 'rb') as f:
        fashionIQ_val_shirt_index_names = pickle.load(f)

    global fashionIQ_test_shirt_index_features
    fashionIQ_test_shirt_index_features = torch.load(
        data_path / 'fashionIQ_test_shirt_index_features.pt', map_location=device).type(data_type).cpu()
    global fashionIQ_test_shirt_index_names
    with open(data_path / 'fashionIQ_test_shirt_index_names.pkl', 'rb') as f:
        fashionIQ_test_shirt_index_names = pickle.load(f)

    global fashionIQ_shirt_index_features
    global fashionIQ_shirt_index_names
    fashionIQ_shirt_index_features = torch.vstack(
        (fashionIQ_val_shirt_index_features, fashionIQ_test_shirt_index_features))
    fashionIQ_shirt_index_names = fashionIQ_val_shirt_index_names + fashionIQ_test_shirt_index_names

    global fashionIQ_val_toptee_index_features
    fashionIQ_val_toptee_index_features = torch.load(
        data_path / 'fashionIQ_val_toptee_index_features.pt', map_location=device).type(data_type).cpu()

    global fashionIQ_val_toptee_index_names
    with open(data_path / 'fashionIQ_val_toptee_index_names.pkl', 'rb') as f:
        fashionIQ_val_toptee_index_names = pickle.load(f)

    global fashionIQ_test_toptee_index_features
    fashionIQ_test_toptee_index_features = torch.load(
        data_path / 'fashionIQ_test_toptee_index_features.pt', map_location=device).type(data_type).cpu()

    global fashionIQ_test_toptee_index_names
    with open(data_path / 'fashionIQ_test_toptee_index_names.pkl', 'rb') as f:
        fashionIQ_test_toptee_index_names = pickle.load(f)

    global fashionIQ_toptee_index_features
    global fashionIQ_toptee_index_names
    fashionIQ_toptee_index_features = torch.vstack(
        (fashionIQ_val_toptee_index_features, fashionIQ_test_toptee_index_features))
    fashionIQ_toptee_index_names = fashionIQ_val_toptee_index_names + fashionIQ_test_toptee_index_names

    global fashion_index_features
    global fashion_index_names
    fashion_index_features = torch.vstack(
        (fashionIQ_dress_index_features, fashionIQ_shirt_index_features, fashionIQ_toptee_index_features))
    fashion_index_names = fashionIQ_dress_index_names + fashionIQ_shirt_index_names + fashionIQ_toptee_index_names


def load_cirr_assets():
    global cirr_val_triplets
    with open(server_base_path / 'cirr_dataset' / 'cirr' / 'captions' / f'cap.rc2.val.json') as f:
        cirr_val_triplets = json.load(f)

    global cirr_name_to_relpath
    with open(server_base_path / 'cirr_dataset' / 'cirr' / 'image_splits' / f'split.rc2.val.json') as f:
        cirr_name_to_relpath = json.load(f)
    with open(server_base_path / 'cirr_dataset' / 'cirr' / 'image_splits' / f'split.rc2.test1.json') as f:
        cirr_name_to_relpath.update(json.load(f))

    global cirr_val_index_features
    cirr_val_index_features = torch.load(data_path / 'cirr_val_index_features.pt', map_location=device).type(
        data_type).cpu()

    global cirr_val_index_names
    with open(data_path / 'cirr_val_index_names.pkl', 'rb') as f:
        cirr_val_index_names = pickle.load(f)

    global cirr_test_index_features
    cirr_test_index_features = torch.load(data_path / 'cirr_test_index_features.pt', map_location=device).type(
        data_type).cpu()

    global cirr_test_index_names
    with open(data_path / 'cirr_test_index_names.pkl', 'rb') as f:
        cirr_test_index_names = pickle.load(f)

    global cirr_index_features
    global cirr_index_names
    cirr_index_features = torch.vstack((cirr_val_index_features, cirr_test_index_features))
    cirr_index_names = cirr_val_index_names + cirr_test_index_names


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
