import os.path as osp
import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
from mmseg.registry import TRANSFORMS

import mmcv
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations
import warnings
import numpy as np


@DATASETS.register_module()
class PascalVOC20Dataset(BaseSegDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """
    METAINFO = dict(
        classes=('aeroplane', 'bicycle', 'bird', 'boat',
                 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                 'sofa', 'train', 'tvmonitor'),
        palette=[[128, 0, 0], [0, 128, 0], [0, 0, 192],
                 [128, 128, 0], [128, 0, 128], [0, 128, 128], [192, 128, 64],
                 [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                 [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                 [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                 [0, 64, 128]])

    def __init__(self,
                 ann_file,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            ann_file=ann_file,
            **kwargs)
        assert fileio.exists(self.data_prefix['img_path'],
                             self.backend_args) and osp.isfile(self.ann_file)


@DATASETS.register_module()
class COCOObjectDataset(BaseSegDataset):
    """
    Implementation borrowed from TCL (https://github.com/kakaobrain/tcl) and GroupViT (https://github.com/NVlabs/GroupViT)
    COCO-Object dataset.
    1 bg class + first 80 classes from the COCO-Stuff dataset.
    """

    METAINFO = dict(

        classes=('background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
                 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'),

        palette=[[0, 0, 0], [0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192], [0, 64, 64], [0, 192, 224],
                 [0, 192, 192], [128, 192, 64], [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224], [0, 0, 64],
                 [0, 160, 192], [128, 0, 96], [128, 0, 192], [0, 32, 192], [128, 128, 224], [0, 0, 192],
                 [128, 160, 192],
                 [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128], [64, 128, 32], [0, 160, 0], [0, 0, 0],
                 [192, 128, 160], [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0], [0, 128, 0], [192, 128, 32],
                 [128, 96, 128], [0, 0, 128], [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160], [0, 96, 128],
                 [128, 128, 128], [64, 0, 160], [128, 224, 128], [128, 128, 64], [192, 0, 32],
                 [128, 96, 0], [128, 0, 192], [0, 128, 32], [64, 224, 0], [0, 0, 64], [128, 128, 160], [64, 96, 0],
                 [0, 128, 192], [0, 128, 160], [192, 224, 0], [0, 128, 64], [128, 128, 32], [192, 32, 128],
                 [0, 64, 192],
                 [0, 0, 32], [64, 160, 128], [128, 64, 64], [128, 0, 160], [64, 32, 128], [128, 192, 192], [0, 0, 160],
                 [192, 160, 128], [128, 192, 0], [128, 0, 96], [192, 32, 0], [128, 64, 128], [64, 128, 96],
                 [64, 160, 0],
                 [0, 64, 0], [192, 128, 224], [64, 32, 0], [0, 192, 128], [64, 128, 224], [192, 160, 0]])

    def __init__(self, **kwargs):
        super(COCOObjectDataset, self).__init__(img_suffix='.jpg', seg_map_suffix='_instanceTrainIds.png', **kwargs)

@DATASETS.register_module()
class COCO80Dataset(BaseSegDataset):
    """
    Implementation borrowed from TCL (https://github.com/kakaobrain/tcl) and GroupViT (https://github.com/NVlabs/GroupViT)
    COCO-Object dataset.
    1 bg class + first 80 classes from the COCO-Stuff dataset.
    """

    METAINFO = dict(

        classes=('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
                 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'),

        palette=[[0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192], [0, 64, 64], [0, 192, 224],
                 [0, 192, 192], [128, 192, 64], [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224], [0, 0, 64],
                 [0, 160, 192], [128, 0, 96], [128, 0, 192], [0, 32, 192], [128, 128, 224], [0, 0, 192],
                 [128, 160, 192],
                 [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128], [64, 128, 32], [0, 160, 0], [0, 0, 0],
                 [192, 128, 160], [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0], [0, 128, 0], [192, 128, 32],
                 [128, 96, 128], [0, 0, 128], [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160], [0, 96, 128],
                 [128, 128, 128], [64, 0, 160], [128, 224, 128], [128, 128, 64], [192, 0, 32],
                 [128, 96, 0], [128, 0, 192], [0, 128, 32], [64, 224, 0], [0, 0, 64], [128, 128, 160], [64, 96, 0],
                 [0, 128, 192], [0, 128, 160], [192, 224, 0], [0, 128, 64], [128, 128, 32], [192, 32, 128],
                 [0, 64, 192],
                 [0, 0, 32], [64, 160, 128], [128, 64, 64], [128, 0, 160], [64, 32, 128], [128, 192, 192], [0, 0, 160],
                 [192, 160, 128], [128, 192, 0], [128, 0, 96], [192, 32, 0], [128, 64, 128], [64, 128, 96],
                 [64, 160, 0],
                 [0, 64, 0], [192, 128, 224], [64, 32, 0], [0, 192, 128], [64, 128, 224], [192, 160, 0]])

    def __init__(self, **kwargs):
        super(COCO80Dataset, self).__init__(img_suffix='.jpg', seg_map_suffix='_instanceTrainIds.png', reduce_zero_label=True, **kwargs)

@DATASETS.register_module()
class PascalContext60Dataset(BaseSegDataset):
    METAINFO = dict(
        classes=('background', 'aeroplane', 'bag', 'bed', 'bedclothes',
                 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle',
                 'building', 'bus', 'cabinet', 'car', 'cat', 'ceiling',
                 'chair', 'cloth', 'computer', 'cow', 'cup', 'curtain', 'dog',
                 'door', 'fence', 'floor', 'flower', 'food', 'grass', 'ground',
                 'horse', 'keyboard', 'light', 'motorbike', 'mountain',
                 'mouse', 'person', 'plate', 'platform', 'pottedplant', 'road',
                 'rock', 'sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow',
                 'sofa', 'table', 'track', 'train', 'tree', 'truck',
                 'tvmonitor', 'wall', 'water', 'window', 'wood'),
        palette=[[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                 [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                 [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                 [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                 [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                 [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                 [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                 [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                 [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                 [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                 [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                 [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                 [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                 [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                 [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255]])

    def __init__(self,
                 ann_file: str,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ann_file=ann_file,
            reduce_zero_label=False,
            **kwargs)


@DATASETS.register_module()
class PascalContext59Dataset(BaseSegDataset):
    METAINFO = dict(
        classes=('aeroplane', 'bag', 'bed', 'bedclothes', 'bench', 'bicycle',
                 'bird', 'boat', 'book', 'bottle', 'building', 'bus',
                 'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth',
                 'computer', 'cow', 'cup', 'curtain', 'dog', 'door', 'fence',
                 'floor', 'flower', 'food', 'grass', 'ground', 'horse',
                 'keyboard', 'light', 'motorbike', 'mountain', 'mouse',
                 'person', 'plate', 'platform', 'pottedplant', 'road', 'rock',
                 'sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa',
                 'table', 'track', 'train', 'tree', 'truck', 'tvmonitor',
                 'wall', 'water', 'window', 'wood'),
        palette=[[180, 120, 120], [6, 230, 230], [80, 50, 50], [4, 200, 3],
                 [120, 120, 80], [140, 140, 140], [204, 5, 255],
                 [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                 [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                 [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                 [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                 [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                 [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                 [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                 [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                 [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                 [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                 [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                 [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                 [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255]])

    def __init__(self,
                 ann_file: str,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ann_file=ann_file,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

@DATASETS.register_module()
class ADE20K847Dataset(BaseSegDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """
    METAINFO = dict(
        classes=("wall", "building, edifice", "sky", "tree", "road, route", "floor, flooring", "ceiling", "bed", "sidewalk, pavement", "earth, ground", "cabinet", "person, individual, someone, somebody, mortal, soul", "grass", "windowpane, window", "car, auto, automobile, machine, motorcar", "mountain, mount", "plant, flora, plant life", "table", "chair", "curtain, drape, drapery, mantle, pall", "door", "sofa, couch, lounge", "sea", "painting, picture", "water", "mirror", "house", "rug, carpet, carpeting", "shelf", "armchair", "fence, fencing", "field", "lamp", "rock, stone", "seat", "river", "desk", "bathtub, bathing tub, bath, tub", "railing, rail", "signboard, sign", "cushion", "path", "work surface", "stairs, steps", "column, pillar", "sink", "wardrobe, closet, press", "snow", "refrigerator, icebox", "base, pedestal, stand", "bridge, span", "blind, screen", "runway", "cliff, drop, drop-off", "sand", "fireplace, hearth, open fireplace", "pillow", "screen door, screen", "toilet, can, commode, crapper, pot, potty, stool, throne", "skyscraper", "grandstand, covered stand", "box", "pool table, billiard table, snooker table", "palm, palm tree", "double door", "coffee table, cocktail table", "counter", "countertop", "chest of drawers, chest, bureau, dresser", "kitchen island", "boat", "waterfall, falls", "stove, kitchen stove, range, kitchen range, cooking stove", "flower", "bookcase", "controls", "book", "stairway, staircase", "streetlight, street lamp", "computer, computing machine, computing device, data processor, electronic computer, information processing system", "bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle", "swivel chair", "light, light source", "bench", "case, display case, showcase, vitrine", "towel", "fountain", "embankment", "television receiver, television, television set, tv, tv set, idiot box, boob tube, telly, goggle box", "van", "hill", "awning, sunshade, sunblind", "poster, posting, placard, notice, bill, card", "truck, motortruck", "airplane, aeroplane, plane", "pole", "tower", "court", "ball", "aircraft carrier, carrier, flattop, attack aircraft carrier", "buffet, counter, sideboard", "hovel, hut, hutch, shack, shanty", "apparel, wearing apparel, dress, clothes", "minibike, motorbike", "animal, animate being, beast, brute, creature, fauna", "chandelier, pendant, pendent", "step, stair", "booth, cubicle, stall, kiosk", "bicycle, bike, wheel, cycle", "doorframe, doorcase", "sconce", "pond", "trade name, brand name, brand, marque", "bannister, banister, balustrade, balusters, handrail", "bag", "traffic light, traffic signal, stoplight", "gazebo", "escalator, moving staircase, moving stairway", "land, ground, soil", "board, plank", "arcade machine", "eiderdown, duvet, continental quilt", "bar", "stall, stand, sales booth", "playground", "ship", "ottoman, pouf, pouffe, puff, hassock", "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin", "bottle", "cradle", "pot, flowerpot", "conveyer belt, conveyor belt, conveyer, conveyor, transporter", "train, railroad train", "stool", "lake", "tank, storage tank", "ice, water ice", "basket, handbasket", "manhole", "tent, collapsible shelter", "canopy", "microwave, microwave oven", "barrel, cask", "dirt track", "beam", "dishwasher, dish washer, dishwashing machine", "plate", "screen, crt screen", "ruins", "washer, automatic washer, washing machine", "blanket, cover", "plaything, toy", "food, solid food", "screen, silver screen, projection screen", "oven", "stage", "beacon, lighthouse, beacon light, pharos", "umbrella", "sculpture", "aqueduct", "container", "scaffolding, staging", "hood, exhaust hood", "curb, curbing, kerb", "roller coaster", "horse, equus caballus", "catwalk", "glass, drinking glass", "vase", "central reservation", "carousel", "radiator", "closet", "machine", "pier, wharf, wharfage, dock", "fan", "inflatable bounce game", "pitch", "paper", "arcade, colonnade", "hot tub", "helicopter", "tray", "partition, divider", "vineyard", "bowl", "bullring", "flag", "pot", "footbridge, overcrossing, pedestrian bridge", "shower", "bag, traveling bag, travelling bag, grip, suitcase", "bulletin board, notice board", "confessional booth", "trunk, tree trunk, bole", "forest", "elevator door", "laptop, laptop computer", "instrument panel", "bucket, pail", "tapestry, tapis", "platform", "jacket", "gate", "monitor, monitoring device", "telephone booth, phone booth, call box, telephone box, telephone kiosk", "spotlight, spot", "ring", "control panel", "blackboard, chalkboard", "air conditioner, air conditioning", "chest", "clock", "sand dune", "pipe, pipage, piping", "vault", "table football", "cannon", "swimming pool, swimming bath, natatorium", "fluorescent, fluorescent fixture", "statue", "loudspeaker, speaker, speaker unit, loudspeaker system, speaker system", "exhibitor", "ladder", "carport", "dam", "pulpit", "skylight, fanlight", "water tower", "grill, grille, grillwork", "display board", "pane, pane of glass, window glass", "rubbish, trash, scrap", "ice rink", "fruit", "patio", "vending machine", "telephone, phone, telephone set", "net", "backpack, back pack, knapsack, packsack, rucksack, haversack", "jar", "track", "magazine", "shutter", "roof", "banner, streamer", "landfill", "post", "altarpiece, reredos", "hat, chapeau, lid", "arch, archway", "table game", "bag, handbag, pocketbook, purse", "document, written document, papers", "dome", "pier", "shanties", "forecourt", "crane", "dog, domestic dog, canis familiaris", "piano, pianoforte, forte-piano", "drawing", "cabin", "ad, advertisement, advertizement, advertising, advertizing, advert", "amphitheater, amphitheatre, coliseum", "monument", "henhouse", "cockpit", "heater, warmer", "windmill, aerogenerator, wind generator", "pool", "elevator, lift", "decoration, ornament, ornamentation", "labyrinth", "text, textual matter", "printer", "mezzanine, first balcony", "mattress", "straw", "stalls", "patio, terrace", "billboard, hoarding", "bus stop", "trouser, pant", "console table, console", "rack", "notebook", "shrine", "pantry", "cart", "steam shovel", "porch", "postbox, mailbox, letter box", "figurine, statuette", "recycling bin", "folding screen", "telescope", "deck chair, beach chair", "kennel", "coffee maker", "altar, communion table, lord's table", "fish", "easel", "artificial golf green", "iceberg", "candlestick, candle holder", "shower stall, shower bath", "television stand", "wall socket, wall plug, electric outlet, electrical outlet, outlet, electric receptacle", "skeleton", "grand piano, grand", "candy, confect", "grille door", "pedestal, plinth, footstall", "jersey, t-shirt, tee shirt", "shoe", "gravestone, headstone, tombstone", "shanty", "structure", "rocking chair, rocker", "bird", "place mat", "tomb", "big top", "gas pump, gasoline pump, petrol pump, island dispenser", "lockers", "cage", "finger", "bleachers", "ferris wheel", "hairdresser chair", "mat", "stands", "aquarium, fish tank, marine museum", "streetcar, tram, tramcar, trolley, trolley car", "napkin, table napkin, serviette", "dummy", "booklet, brochure, folder, leaflet, pamphlet", "sand trap", "shop, store", "table cloth", "service station", "coffin", "drawer", "cages", "slot machine, coin machine", "balcony", "volleyball court", "table tennis", "control table", "shirt", "merchandise, ware, product", "railway", "parterre", "chimney", "can, tin, tin can", "tanks", "fabric, cloth, material, textile", "alga, algae", "system", "map", "greenhouse", "mug", "barbecue", "trailer", "toilet tissue, toilet paper, bathroom tissue", "organ", "dishrag, dishcloth", "island", "keyboard", "trench", "basket, basketball hoop, hoop", "steering wheel, wheel", "pitcher, ewer", "goal", "bread, breadstuff, staff of life", "beds", "wood", "file cabinet", "newspaper, paper", "motorboat", "rope", "guitar", "rubble", "scarf", "barrels", "cap", "leaves", "control tower", "dashboard", "bandstand", "lectern", "switch, electric switch, electrical switch", "baseboard, mopboard, skirting board", "shower room", "smoke", "faucet, spigot", "bulldozer", "saucepan", "shops", "meter", "crevasse", "gear", "candelabrum, candelabra", "sofa bed", "tunnel", "pallet", "wire, conducting wire", "kettle, boiler", "bidet", "baby buggy, baby carriage, carriage, perambulator, pram, stroller, go-cart, pushchair, pusher", "music stand", "pipe, tube", "cup", "parking meter", "ice hockey rink", "shelter", "weeds", "temple", "patty, cake", "ski slope", "panel", "wallet", "wheel", "towel rack, towel horse", "roundabout", "canister, cannister, tin", "rod", "soap dispenser", "bell", "canvas", "box office, ticket office, ticket booth", "teacup", "trellis", "workbench", "valley, vale", "toaster", "knife", "podium", "ramp", "tumble dryer", "fireplug, fire hydrant, plug", "gym shoe, sneaker, tennis shoe", "lab bench", "equipment", "rocky formation", "plastic", "calendar", "caravan", "check-in-desk", "ticket counter", "brush", "mill", "covered bridge", "bowling alley", "hanger", "excavator", "trestle", "revolving door", "blast furnace", "scale, weighing machine", "projector", "soap", "locker", "tractor", "stretcher", "frame", "grating", "alembic", "candle, taper, wax light", "barrier", "cardboard", "cave", "puddle", "tarp", "price tag", "watchtower", "meters", "light bulb, lightbulb, bulb, incandescent lamp, electric light, electric-light bulb", "tracks", "hair dryer", "skirt", "viaduct", "paper towel", "coat", "sheet", "fire extinguisher, extinguisher, asphyxiator", "water wheel", "pottery, clayware", "magazine rack", "teapot", "microphone, mike", "support", "forklift", "canyon", "cash register, register", "leaf, leafage, foliage", "remote control, remote", "soap dish", "windshield, windscreen", "cat", "cue, cue stick, pool cue, pool stick", "vent, venthole, vent-hole, blowhole", "videos", "shovel", "eaves", "antenna, aerial, transmitting aerial", "shipyard", "hen, biddy", "traffic cone", "washing machines", "truck crane", "cds", "niche", "scoreboard", "briefcase", "boot", "sweater, jumper", "hay", "pack", "bottle rack", "glacier", "pergola", "building materials", "television camera", "first floor", "rifle", "tennis table", "stadium", "safety belt", "cover", "dish rack", "synthesizer", "pumpkin", "gutter", "fruit stand", "ice floe, floe", "handle, grip, handgrip, hold", "wheelchair", "mousepad, mouse mat", "diploma", "fairground ride", "radio", "hotplate", "junk", "wheelbarrow", "stream", "toll plaza", "punching bag", "trough", "throne", "chair desk", "weighbridge", "extractor fan", "hanging clothes", "dish, dish aerial, dish antenna, saucer", "alarm clock, alarm", "ski lift", "chain", "garage", "mechanical shovel", "wine rack", "tramway", "treadmill", "menu", "block", "well", "witness stand", "branch", "duck", "casserole", "frying pan", "desk organizer", "mast", "spectacles, specs, eyeglasses, glasses", "service elevator", "dollhouse", "hammock", "clothes hanging", "photocopier", "notepad", "golf cart", "footpath", "cross", "baptismal font", "boiler", "skip", "rotisserie", "tables", "water mill", "helmet", "cover curtain", "brick", "table runner", "ashtray", "street box", "stick", "hangers", "cells", "urinal", "centerpiece", "portable fridge", "dvds", "golf club", "skirting board", "water cooler", "clipboard", "camera, photographic camera", "pigeonhole", "chips", "food processor", "post box", "lid", "drum", "blender", "cave entrance", "dental chair", "obelisk", "canoe", "mobile", "monitors", "pool ball", "cue rack", "baggage carts", "shore", "fork", "paper filer", "bicycle rack", "coat rack", "garland", "sports bag", "fish tank", "towel dispenser", "carriage", "brochure", "plaque", "stringer", "iron", "spoon", "flag pole", "toilet brush", "book stand", "water faucet, water tap, tap, hydrant", "ticket office", "broom", "dvd", "ice bucket", "carapace, shell, cuticle, shield", "tureen", "folders", "chess", "root", "sewing machine", "model", "pen", "violin", "sweatshirt", "recycling materials", "mitten", "chopping board, cutting board", "mask", "log", "mouse, computer mouse", "grill", "hole", "target", "trash bag", "chalk", "sticks", "balloon", "score", "hair spray", "roll", "runner", "engine", "inflatable glove", "games", "pallets", "baskets", "coop", "dvd player", "rocking horse", "buckets", "bread rolls", "shawl", "watering can", "spotlights", "post-it", "bowls", "security camera", "runner cloth", "lock", "alarm, warning device, alarm system", "side", "roulette", "bone", "cutlery", "pool balls", "wheels", "spice rack", "plant pots", "towel ring", "bread box", "video", "funfair", "breads", "tripod", "ironing board", "skimmer", "hollow", "scratching post", "tricycle", "file box", "mountain pass", "tombstones", "cooker", "card game, cards", "golf bag", "towel paper", "chaise lounge", "sun", "toilet paper holder", "rake", "key", "umbrella stand", "dartboard", "transformer", "fireplace utensils", "sweatshirts", "cellular telephone, cellular phone, cellphone, cell, mobile phone", "tallboy", "stapler", "sauna", "test tube", "palette", "shopping carts", "tools", "push button, push, button", "star", "roof rack", "barbed wire", "spray", "ear", "sponge", "racket", "tins", "eyeglasses", "file", "scarfs", "sugar bowl", "flip flop", "headstones", "laptop bag", "leash", "climbing frame", "suit hanger", "floor spotlight", "plate rack", "sewer", "hard drive", "sprinkler", "tools box", "necklace", "bulbs", "steel industry", "club", "jack", "door bars", "control panel, instrument panel, control board, board, panel", "hairbrush", "napkin holder", "office", "smoke detector", "utensils", "apron", "scissors", "terminal", "grinder", "entry phone", "newspaper stand", "pepper shaker", "onions", "central processing unit, cpu, c p u , central processor, processor, mainframe", "tape", "bat", "coaster", "calculator", "potatoes", "luggage rack", "salt", "street number", "viewpoint", "sword", "cd", "rowing machine", "plug", "andiron, firedog, dog, dog-iron", "pepper", "tongs", "bonfire", "dog dish", "belt", "dumbbells", "videocassette recorder, vcr", "hook", "envelopes", "shower faucet", "watch", "padlock", "swimming pool ladder", "spanners", "gravy boat", "notice board", "trash bags", "fire alarm", "ladle", "stethoscope", "rocket", "funnel", "bowling pins", "valve", "thermometer", "cups", "spice jar", "night light", "soaps", "games table", "slotted spoon", "reel", "scourer", "sleeping robe", "desk mat", "dumbbell", "hammer", "tie", "typewriter", "shaker", "cheese dish", "sea star", "racquet", "butane gas cylinder", "paper weight", "shaving brush", "sunglasses", "gear shift", "towel rail", "adding machine, totalizer, totaliser"),
        # palette=[[128, 0, 0], [0, 128, 0], [0, 0, 192],
        #          [128, 128, 0], [128, 0, 128], [0, 128, 128], [192, 128, 64],
        #          [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
        #          [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
        #          [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
        #          [0, 64, 128]])
    )

    def __init__(self,
                 # ann_file,
                 img_suffix='.jpg',
                 seg_map_suffix='.tif',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            ignore_index=65535,
            # ann_file=ann_file,
            **kwargs)
        # assert fileio.exists(self.data_prefix['img_path'],
        #                      self.backend_args) and osp.isfile(self.ann_file)

@DATASETS.register_module()
class PascalContext459Dataset(BaseSegDataset):
    METAINFO = dict(
        classes=("accordion", "aeroplane", "airconditioner", "antenna", "artillery", "ashtray", "atrium",
                 "babycarriage", "bag", "ball", "balloon", "bambooweaving", "barrel", "baseballbat", "basket",
                 "basketballbackboard", "bathtub", "bed", "bedclothes", "beer", "bell", "bench", "bicycle",
                 "binoculars",
                 "bird", "birdcage", "birdfeeder", "birdnest", "blackboard", "board", "boat", "bone", "book", "bottle",
                 "bottleopener", "bowl", "box", "bracelet", "brick", "bridge", "broom", "brush", "bucket", "building",
                 "bus", "cabinet", "cabinetdoor", "cage", "cake", "calculator", "calendar", "camel", "camera",
                 "cameralens", "can", "candle", "candleholder", "cap", "car", "card", "cart", "case", "casetterecorder",
                 "cashregister", "cat", "cd", "cdplayer", "ceiling", "cellphone", "cello", "chain", "chair",
                 "chessboard",
                 "chicken", "chopstick", "clip", "clippers", "clock", "closet", "cloth", "clothestree", "coffee",
                 "coffeemachine", "comb", "computer", "concrete", "cone", "container", "controlbooth", "controller",
                 "cooker", "copyingmachine", "coral", "cork", "corkscrew", "counter", "court", "cow", "crabstick",
                 "crane", "crate", "cross", "crutch", "cup", "curtain", "cushion", "cuttingboard", "dais", "disc",
                 "disccase", "dishwasher", "dock", "dog", "dolphin", "door", "drainer", "dray", "drinkdispenser",
                 "drinkingmachine", "drop", "drug", "drum", "drumkit", "duck", "dumbbell", "earphone", "earrings",
                 "egg", "electricfan", "electriciron", "electricpot", "electricsaw", "electronickeyboard", "engine",
                 "envelope", "equipment", "escalator", "exhibitionbooth", "extinguisher", "eyeglass", "fan", "faucet",
                 "faxmachine", "fence", "ferriswheel", "fireextinguisher", "firehydrant", "fireplace", "fish",
                 "fishtank",
                 "fishbowl", "fishingnet", "fishingpole", "flag", "flagstaff", "flame", "flashlight", "floor", "flower",
                 "fly", "foam", "food", "footbridge", "forceps", "fork", "forklift", "fountain", "fox", "frame",
                 "fridge",
                 "frog", "fruit", "funnel", "furnace", "gamecontroller", "gamemachine", "gascylinder", "gashood",
                 "gasstove",
                 "giftbox", "glass", "glassmarble", "globe", "glove", "goal", "grandstand", "grass", "gravestone",
                 "ground",
                 "guardrail", "guitar", "gun", "hammer", "handcart", "handle", "handrail", "hanger", "harddiskdrive",
                 "hat", "hay", "headphone", "heater", "helicopter", "helmet", "holder", "hook", "horse",
                 "horse-drawncarriage",
                 "hot-airballoon", "hydrovalve", "ice", "inflatorpump", "ipod", "iron", "ironingboard", "jar", "kart",
                 "kettle", "key", "keyboard", "kitchenrange", "kite", "knife", "knifeblock", "ladder", "laddertruck",
                 "ladle", "laptop", "leaves", "lid", "lifebuoy", "light", "lightbulb", "lighter", "line", "lion",
                 "lobster",
                 "lock", "machine", "mailbox", "mannequin", "map", "mask", "mat", "matchbook", "mattress", "menu",
                 "metal",
                 "meterbox", "microphone", "microwave", "mirror", "missile", "model", "money", "monkey", "mop",
                 "motorbike",
                 "mountain", "mouse", "mousepad", "musicalinstrument", "napkin", "net", "newspaper", "oar", "ornament",
                 "outlet", "oven", "oxygenbottle", "pack", "pan", "paper", "paperbox", "papercutter", "parachute",
                 "parasol",
                 "parterre", "patio", "pelage", "pen", "pencontainer", "pencil", "person", "photo", "piano", "picture",
                 "pig",
                 "pillar", "pillow", "pipe", "pitcher", "plant", "plastic", "plate", "platform", "player", "playground",
                 "pliers",
                 "plume", "poker", "pokerchip", "pole", "pooltable", "postcard", "poster", "pot", "pottedplant",
                 "printer", "projector",
                 "pumpkin", "rabbit", "racket", "radiator", "radio", "rail", "rake", "ramp", "rangehood", "receiver",
                 "recorder",
                 "recreationalmachines", "remotecontrol", "road", "robot", "rock", "rocket", "rockinghorse", "rope",
                 "rug", "ruler",
                 "runway", "saddle", "sand", "saw", "scale", "scanner", "scissors", "scoop", "screen", "screwdriver",
                 "sculpture",
                 "scythe", "sewer", "sewingmachine", "shed", "sheep", "shell", "shelves", "shoe", "shoppingcart",
                 "shovel", "sidecar",
                 "sidewalk", "sign", "signallight", "sink", "skateboard", "ski", "sky", "sled", "slippers", "smoke",
                 "snail", "snake",
                 "snow", "snowmobiles", "sofa", "spanner", "spatula", "speaker", "speedbump", "spicecontainer", "spoon",
                 "sprayer",
                 "squirrel", "stage", "stair", "stapler", "stick", "stickynote", "stone", "stool", "stove", "straw",
                 "stretcher", "sun",
                 "sunglass", "sunshade", "surveillancecamera", "swan", "sweeper", "swimring", "swimmingpool", "swing",
                 "switch", "table",
                 "tableware", "tank", "tap", "tape", "tarp", "telephone", "telephonebooth", "tent", "tire", "toaster",
                 "toilet", "tong",
                 "tool", "toothbrush", "towel", "toy", "toycar", "track", "train", "trampoline", "trashbin", "tray",
                 "tree", "tricycle",
                 "tripod", "trophy", "truck", "tube", "turtle", "tvmonitor", "tweezers", "typewriter", "umbrella",
                 "unknown", "vacuumcleaner",
                 "vendingmachine", "videocamera", "videogameconsole", "videoplayer", "videotape", "violin", "wakeboard",
                 "wall", "wallet",
                 "wardrobe", "washingmachine", "watch", "water", "waterdispenser", "waterpipe", "waterskateboard",
                 "watermelon", "whale",
                 "wharf", "wheel", "wheelchair", "window", "windowblinds", "wineglass", "wire", "wood", "wool"),
    )

    def __init__(self,
                 ann_file,
                 img_suffix='.jpg',
                 seg_map_suffix='.tif',
                 reduce_zero_label=False,
                 **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ann_file=ann_file,
            reduce_zero_label=reduce_zero_label,
            **kwargs)


@TRANSFORMS.register_module()
class MyLoadAnnotations(MMCV_LoadAnnotations):
    """Load annotations for semantic segmentation provided by dataset.

    The annotation format is as the following:

    .. code-block:: python

        {
            # Filename of semantic segmentation ground truth file.
            'seg_map_path': 'a/b/c'
        }

    After this module, the annotation has been changed to the format below:

    .. code-block:: python

        {
            # in str
            'seg_fields': List
             # In uint8 type.
            'gt_seg_map': np.ndarray (H, W)
        }

    Required Keys:

    - seg_map_path (str): Path of semantic segmentation ground truth file.

    Added Keys:

    - seg_fields (List)
    - gt_seg_map (np.uint8)

    Args:
        reduce_zero_label (bool, optional): Whether reduce all label value
            by 1. Usually used for datasets where 0 is background label.
            Defaults to None.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :fun:``mmcv.imfrombytes`` for details.
            Defaults to 'pillow'.
        backend_args (dict): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(
            self,
            reduce_zero_label=None,
            backend_args=None,
            imdecode_backend='pillow',
    ) -> None:
        super().__init__(
            with_bbox=False,
            with_label=False,
            with_seg=True,
            with_keypoints=False,
            imdecode_backend=imdecode_backend,
            backend_args=backend_args)
        self.reduce_zero_label = reduce_zero_label
        if self.reduce_zero_label is not None:
            warnings.warn('`reduce_zero_label` will be deprecated, '
                          'if you would like to ignore the zero label, please '
                          'set `reduce_zero_label=True` when dataset '
                          'initialized')
        self.imdecode_backend = imdecode_backend

    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        img_bytes = fileio.get(
            results['seg_map_path'], backend_args=self.backend_args)
        gt_semantic_seg = mmcv.imfrombytes(
            img_bytes, flag='unchanged',
            backend=self.imdecode_backend).squeeze().astype(np.uint16)

        # reduce zero_label
        if self.reduce_zero_label is None:
            self.reduce_zero_label = results['reduce_zero_label']
        assert self.reduce_zero_label == results['reduce_zero_label'], \
            'Initialize dataset with `reduce_zero_label` as ' \
            f'{results["reduce_zero_label"]} but when load annotation ' \
            f'the `reduce_zero_label` is {self.reduce_zero_label}'
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        results['gt_seg_map'] = gt_semantic_seg
        results['seg_fields'].append('gt_seg_map')

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str

# ===================== MESS benchmark datasets (Table 3) =====================

@DATASETS.register_module()
class FoodSeg103(BaseSegDataset):
    METAINFO = dict(
        classes=(
            'background', 'candy', 'egg tart', 'french fries', 'chocolate', 'biscuit', 'popcorn', 'pudding', 'ice cream', 'cheese butter', 'cake', 'wine', 'milkshake', 'coffee',
            'juice', 'milk', 'tea',
            'almond', 'red beans', 'cashew', 'dried cranberries', 'soy', 'walnut', 'peanut', 'egg', 'apple', 'date', 'apricot', 'avocado', 'banana', 'strawberry', 'cherry',
            'blueberry', 'raspberry',
            'mango', 'olives', 'peach', 'lemon', 'pear', 'fig', 'pineapple', 'grape', 'kiwi', 'melon', 'orange', 'watermelon', 'steak', 'pork', 'chicken duck', 'sausage',
            'fried meat', 'lamb', 'sauce',
            'crab', 'fish', 'shellfish', 'shrimp', 'soup', 'bread', 'corn', 'hamburg', 'pizza', 'hanamaki baozi', 'wonton dumplings', 'pasta', 'noodles', 'rice', 'pie', 'tofu',
            'eggplant', 'potato',
            'garlic', 'cauliflower', 'tomato', 'kelp', 'seaweed', 'spring onion', 'rape', 'ginger', 'okra', 'lettuce', 'pumpkin', 'cucumber', 'white radish', 'carrot', 'asparagus',
            'bamboo shoots',
            'broccoli', 'celery stick', 'cilantro mint', 'snow peas', 'cabbage', 'bean sprouts', 'onion', 'pepper', 'green beans', 'French beans', 'king oyster mushroom',
            'shiitake', 'enoki mushroom',
            'oyster mushroom', 'white button mushroom', 'salad', 'other ingredients'),
        palette=[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9], [10, 10, 10], [11, 11, 11], [12, 12, 12],
                 [13, 13, 13], [14, 14, 14],
                 [15, 15, 15], [16, 16, 16], [17, 17, 17], [18, 18, 18], [19, 19, 19], [20, 20, 20], [21, 21, 21], [22, 22, 22], [23, 23, 23], [24, 24, 24], [25, 25, 25],
                 [26, 26, 26], [27, 27, 27],
                 [28, 28, 28], [29, 29, 29], [30, 30, 30], [31, 31, 31], [32, 32, 32], [33, 33, 33], [34, 34, 34], [35, 35, 35], [36, 36, 36], [37, 37, 37], [38, 38, 38],
                 [39, 39, 39], [40, 40, 40],
                 [41, 41, 41], [42, 42, 42], [43, 43, 43], [44, 44, 44], [45, 45, 45], [46, 46, 46], [47, 47, 47], [48, 48, 48], [49, 49, 49], [50, 50, 50], [51, 51, 51],
                 [52, 52, 52], [53, 53, 53],
                 [54, 54, 54], [55, 55, 55], [56, 56, 56], [57, 57, 57], [58, 58, 58], [59, 59, 59], [60, 60, 60], [61, 61, 61], [62, 62, 62], [63, 63, 63], [64, 64, 64],
                 [65, 65, 65], [66, 66, 66],
                 [67, 67, 67], [68, 68, 68], [69, 69, 69], [70, 70, 70], [71, 71, 71], [72, 72, 72], [73, 73, 73], [74, 74, 74], [75, 75, 75], [76, 76, 76], [77, 77, 77],
                 [78, 78, 78], [79, 79, 79],
                 [80, 80, 80], [81, 81, 81], [82, 82, 82], [83, 83, 83], [84, 84, 84], [85, 85, 85], [86, 86, 86], [87, 87, 87], [88, 88, 88], [89, 89, 89], [90, 90, 90],
                 [91, 91, 91], [92, 92, 92],
                 [93, 93, 93], [94, 94, 94], [95, 95, 95], [96, 96, 96], [97, 97, 97], [98, 98, 98], [99, 99, 99], [100, 100, 100], [101, 101, 101], [102, 102, 102],
                 [103, 103, 103]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 ignore_index=255,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ignore_index=ignore_index,
            reduce_zero_label=False,
            **kwargs)


@DATASETS.register_module()
class DRAM(BaseSegDataset):
    METAINFO = dict(
        classes=(
            ['bird', 'boat', 'bottle', 'cat', 'chair', 'cow', 'dog', 'horse', 'person', 'potted-plant', 'sheep', 'background']),
        palette=[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9], [10, 10, 10], [11, 11, 11]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 ignore_index=255,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ignore_index=ignore_index,
            reduce_zero_label=False,
            **kwargs)


@DATASETS.register_module()
class ATLANTIS(BaseSegDataset):
    METAINFO = dict(
        classes=(
            ['bicycle', 'boat', 'breakwater', 'bridge', 'building', 'bus', 'canal', 'car', 'cliff', 'culvert', 'cypress tree', 'dam', 'ditch', 'fence', 'fire hydrant', 'fjord',
             'flood', 'glaciers',
             'hot spring', 'lake', 'levee', 'lighthouse', 'mangrove', 'marsh', 'motorcycle', 'offshore platform', 'parking meter', 'person', 'pier', 'pipeline', 'pole', 'puddle',
             'rapids',
             'reservoir', 'river', 'river delta', 'road', 'sea', 'ship', 'shoreline', 'sidewalk', 'sky', 'snow', 'spillway', 'swimming pool', 'terrain', 'traffic sign', 'train',
             'truck', 'umbrella',
             'vegetation', 'wall', 'water tower', 'water well', 'waterfall', 'wetland']),
        palette=[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9], [10, 10, 10], [11, 11, 11], [12, 12, 12],
                 [13, 13, 13], [14, 14, 14],
                 [15, 15, 15], [16, 16, 16], [17, 17, 17], [18, 18, 18], [19, 19, 19], [20, 20, 20], [21, 21, 21], [22, 22, 22], [23, 23, 23], [24, 24, 24], [25, 25, 25],
                 [26, 26, 26], [27, 27, 27],
                 [28, 28, 28], [29, 29, 29], [30, 30, 30], [31, 31, 31], [32, 32, 32], [33, 33, 33], [34, 34, 34], [35, 35, 35], [36, 36, 36], [37, 37, 37], [38, 38, 38],
                 [39, 39, 39], [40, 40, 40],
                 [41, 41, 41], [42, 42, 42], [43, 43, 43], [44, 44, 44], [45, 45, 45], [46, 46, 46], [47, 47, 47], [48, 48, 48], [49, 49, 49], [50, 50, 50], [51, 51, 51],
                 [52, 52, 52], [53, 53, 53],
                 [54, 54, 54], [55, 55, 55]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 ignore_index=255,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ignore_index=ignore_index,
            reduce_zero_label=False,
            **kwargs)


@DATASETS.register_module()
class SUIM(BaseSegDataset):
    METAINFO = dict(
        classes=(
            ['background (waterbody)', 'human diver', 'aquatic plants and sea-grass', 'wrecks and ruins', 'robots (auvs/rovs/instruments)', 'reefs and invertebrates',
             'fish and vertebrates',
             'sea-floor and rocks']),
        palette=[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 ignore_index=255,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ignore_index=ignore_index,
            reduce_zero_label=False,
            **kwargs)


@DATASETS.register_module()
class CWFID(BaseSegDataset):
    METAINFO = dict(
        classes=(
            ['ground', 'crop seedling', 'weed']),
        palette=[[0, 0, 0], [1, 1, 1], [2, 2, 2]])

    def __init__(self,
                 img_suffix='_image.png',
                 seg_map_suffix='_annotation.png',
                 ignore_index=255,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ignore_index=ignore_index,
            reduce_zero_label=False,
            **kwargs)


@DATASETS.register_module()
class CUB(BaseSegDataset):
    METAINFO = dict(
        classes=(
            ['background', 'Black-footed Albatross', 'Laysan Albatross', 'Sooty Albatross', 'Groove billed Ani', 'Crested Auklet', 'Least Auklet', 'Parakeet Auklet',
             'Rhinoceros Auklet',
             'Brewer Blackbird', 'Red winged Blackbird', 'Rusty Blackbird', 'Yellow-headed Blackbird', 'Bobolink', 'Indigo Bunting', 'Lazuli Bunting', 'Painted Bunting',
             'Cardinal', 'Spotted Catbird',
             'Gray Catbird', 'Yellow-breasted Chat', 'Eastern Towhee', 'Chuck will Widow', 'Brandt Cormorant', 'Red-faced Cormorant', 'Pelagic Cormorant', 'Bronzed Cowbird',
             'Shiny Cowbird',
             'Brown Creeper', 'American Crow', 'Fish Crow', 'Black-billed Cuckoo', 'Mangrove Cuckoo', 'Yellow-billed Cuckoo', 'Gray-crowned Rosy-Finch', 'Purple Finch',
             'Northern Flicker',
             'Acadian Flycatcher', 'Great Crested Flycatcher', 'Least Flycatcher', 'Olive-sided Flycatcher', 'Scissor-tailed Flycatcher', 'Vermilion Flycatcher',
             'Yellow-bellied Flycatcher',
             'Frigatebird', 'Northern Fulmar', 'Gadwall', 'American Goldfinch', 'European Goldfinch', 'Boat-tailed Grackle', 'Eared Grebe', 'Horned Grebe', 'Pied-billed Grebe',
             'Western Grebe',
             'Blue Grosbeak', 'Evening Grosbeak', 'Pine Grosbeak', 'Rose-breasted Grosbeak', 'Pigeon Guillemot', 'California Gull', 'Glaucous-winged Gull', 'Heermann Gull',
             'Herring Gull',
             'Ivory Gull', 'Ring-billed Gull', 'Slaty-backed Gull', 'Western Gull', 'Anna Hummingbird', 'Ruby-throated Hummingbird', 'Rufous Hummingbird', 'Green Violetear',
             'Long-tailed Jaeger',
             'Pomarine Jaeger', 'Blue Jay', 'Florida Jay', 'Green Jay', 'Dark-eyed Junco', 'Tropical Kingbird', 'Gray Kingbird', 'Belted Kingfisher', 'Green Kingfisher',
             'Pied Kingfisher',
             'Ringed Kingfisher', 'White-breasted Kingfisher', 'Red-legged Kittiwake', 'Horned Lark', 'Pacific Loon', 'Mallard', 'Western Meadowlark', 'Hooded Merganser',
             'Red-breasted Merganser',
             'Mockingbird', 'Nighthawk', 'Clark Nutcracker', 'White-breasted Nuthatch', 'Baltimore Oriole', 'Hooded Oriole', 'Orchard Oriole', 'Scott Oriole', 'Ovenbird',
             'Brown Pelican',
             'White Pelican', 'Western Wood-Pewee', 'Sayornis', 'American Pipit', 'Whip-poor-Will', 'Horned Puffin', 'Common Raven', 'White-necked Raven', 'American Redstart',
             'Geococcyx',
             'Loggerhead Shrike', 'Great Grey Shrike', 'Baird Sparrow', 'Black-throated Sparrow', 'Brewer Sparrow', 'Chipping Sparrow', 'Clay-colored Sparrow', 'House Sparrow',
             'Field Sparrow',
             'Fox Sparrow', 'Grasshopper Sparrow', 'Harris Sparrow', 'Henslow Sparrow', 'Le Conte Sparrow', 'Lincoln Sparrow', 'Nelson Sharp-tailed Sparrow', 'Savannah Sparrow',
             'Seaside Sparrow',
             'Song Sparrow', 'Tree Sparrow', 'Vesper Sparrow', 'White-crowned Sparrow', 'White-throated Sparrow', 'Cape Glossy Starling', 'Bank Swallow', 'Barn Swallow',
             'Cliff Swallow',
             'Tree Swallow', 'Scarlet Tanager', 'Summer Tanager', 'Artic Tern', 'Black Tern', 'Caspian Tern', 'Common Tern', 'Elegant Tern', 'Forsters Tern', 'Least Tern',
             'Green-tailed Towhee',
             'Brown Thrasher', 'Sage Thrasher', 'Black-capped Vireo', 'Blue-headed Vireo', 'Philadelphia Vireo', 'Red-eyed Vireo', 'Warbling Vireo', 'White-eyed Vireo',
             'Yellow-throated Vireo',
             'Bay-breasted Warbler', 'Black-and-white Warbler', 'Black-throated Blue Warbler', 'Blue-winged Warbler', 'Canada Warbler', 'Cape May Warbler', 'Cerulean Warbler',
             'Chestnut-sided Warbler', 'Golden-winged Warbler', 'Hooded Warbler', 'Kentucky Warbler', 'Magnolia Warbler', 'Mourning Warbler', 'Myrtle Warbler', 'Nashville Warbler',
             'Orange-crowned Warbler', 'Palm Warbler', 'Pine Warbler', 'Prairie Warbler', 'Prothonotary Warbler', 'Swainson Warbler', 'Tennessee Warbler', 'Wilson Warbler',
             'Worm-eating Warbler',
             'Yellow Warbler', 'Northern Waterthrush', 'Louisiana Waterthrush', 'Bohemian Waxwing', 'Cedar Waxwing', 'American Three-toed Woodpecker', 'Pileated Woodpecker',
             'Red-bellied Woodpecker',
             'Red-cockaded Woodpecker', 'Red-headed Woodpecker', 'Downy Woodpecker', 'Bewick Wren', 'Cactus Wren', 'Carolina Wren', 'House Wren', 'Marsh Wren', 'Rock Wren',
             'Winter Wren',
             'Common Yellowthroat']),
        palette=[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9], [10, 10, 10], [11, 11, 11], [12, 12, 12],
                 [13, 13, 13], [14, 14, 14],
                 [15, 15, 15], [16, 16, 16], [17, 17, 17], [18, 18, 18], [19, 19, 19], [20, 20, 20], [21, 21, 21], [22, 22, 22], [23, 23, 23], [24, 24, 24], [25, 25, 25],
                 [26, 26, 26], [27, 27, 27],
                 [28, 28, 28], [29, 29, 29], [30, 30, 30], [31, 31, 31], [32, 32, 32], [33, 33, 33], [34, 34, 34], [35, 35, 35], [36, 36, 36], [37, 37, 37], [38, 38, 38],
                 [39, 39, 39], [40, 40, 40],
                 [41, 41, 41], [42, 42, 42], [43, 43, 43], [44, 44, 44], [45, 45, 45], [46, 46, 46], [47, 47, 47], [48, 48, 48], [49, 49, 49], [50, 50, 50], [51, 51, 51],
                 [52, 52, 52], [53, 53, 53],
                 [54, 54, 54], [55, 55, 55], [56, 56, 56], [57, 57, 57], [58, 58, 58], [59, 59, 59], [60, 60, 60], [61, 61, 61], [62, 62, 62], [63, 63, 63], [64, 64, 64],
                 [65, 65, 65], [66, 66, 66],
                 [67, 67, 67], [68, 68, 68], [69, 69, 69], [70, 70, 70], [71, 71, 71], [72, 72, 72], [73, 73, 73], [74, 74, 74], [75, 75, 75], [76, 76, 76], [77, 77, 77],
                 [78, 78, 78], [79, 79, 79],
                 [80, 80, 80], [81, 81, 81], [82, 82, 82], [83, 83, 83], [84, 84, 84], [85, 85, 85], [86, 86, 86], [87, 87, 87], [88, 88, 88], [89, 89, 89], [90, 90, 90],
                 [91, 91, 91], [92, 92, 92],
                 [93, 93, 93], [94, 94, 94], [95, 95, 95], [96, 96, 96], [97, 97, 97], [98, 98, 98], [99, 99, 99], [100, 100, 100], [101, 101, 101], [102, 102, 102],
                 [103, 103, 103], [104, 104, 104],
                 [105, 105, 105], [106, 106, 106], [107, 107, 107], [108, 108, 108], [109, 109, 109], [110, 110, 110], [111, 111, 111], [112, 112, 112], [113, 113, 113],
                 [114, 114, 114],
                 [115, 115, 115], [116, 116, 116], [117, 117, 117], [118, 118, 118], [119, 119, 119], [120, 120, 120], [121, 121, 121], [122, 122, 122], [123, 123, 123],
                 [124, 124, 124],
                 [125, 125, 125], [126, 126, 126], [127, 127, 127], [128, 128, 128], [129, 129, 129], [130, 130, 130], [131, 131, 131], [132, 132, 132], [133, 133, 133],
                 [134, 134, 134],
                 [135, 135, 135], [136, 136, 136], [137, 137, 137], [138, 138, 138], [139, 139, 139], [140, 140, 140], [141, 141, 141], [142, 142, 142], [143, 143, 143],
                 [144, 144, 144],
                 [145, 145, 145], [146, 146, 146], [147, 147, 147], [148, 148, 148], [149, 149, 149], [150, 150, 150], [151, 151, 151], [152, 152, 152], [153, 153, 153],
                 [154, 154, 154],
                 [155, 155, 155], [156, 156, 156], [157, 157, 157], [158, 158, 158], [159, 159, 159], [160, 160, 160], [161, 161, 161], [162, 162, 162], [163, 163, 163],
                 [164, 164, 164],
                 [165, 165, 165], [166, 166, 166], [167, 167, 167], [168, 168, 168], [169, 169, 169], [170, 170, 170], [171, 171, 171], [172, 172, 172], [173, 173, 173],
                 [174, 174, 174],
                 [175, 175, 175], [176, 176, 176], [177, 177, 177], [178, 178, 178], [179, 179, 179], [180, 180, 180], [181, 181, 181], [182, 182, 182], [183, 183, 183],
                 [184, 184, 184],
                 [185, 185, 185], [186, 186, 186], [187, 187, 187], [188, 188, 188], [189, 189, 189], [190, 190, 190], [191, 191, 191], [192, 192, 192], [193, 193, 193],
                 [194, 194, 194],
                 [195, 195, 195], [196, 196, 196], [197, 197, 197], [198, 198, 198], [199, 199, 199], [200, 200, 200]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 ignore_index=255,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ignore_index=ignore_index,
            reduce_zero_label=False,
            **kwargs)


@DATASETS.register_module()
class MHPDATASET(BaseSegDataset):
    METAINFO = dict(
        classes=(
            ['others', 'hat', 'hair', 'sunglasses', 'upper clothes', 'skirt', 'pants', 'dress', 'belt', 'left shoe',
             'right shoe', 'face', 'left leg', 'right leg', 'left arm', 'right arm', 'bag', 'scarf', 'torso skin']),
        palette=[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9], [10, 10, 10], [11, 11, 11], [12, 12, 12],
                 [13, 13, 13], [14, 14, 14],
                 [15, 15, 15], [16, 16, 16], [17, 17, 17], [18, 18, 18]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 ignore_index=255,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ignore_index=ignore_index,
            reduce_zero_label=False,
            **kwargs)


@DATASETS.register_module()
class DARK_SURICHDATASET(BaseSegDataset):
    METAINFO = dict(
        classes=(
            ['unlabeled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign',
             'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
             'motorcycle', 'bicycle']),
        palette=[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9], [10, 10, 10], [11, 11, 11], [12, 12, 12],
                 [13, 13, 13], [14, 14, 14],
                 [15, 15, 15], [16, 16, 16], [17, 17, 17], [18, 18, 18], [19, 19, 19]])

    def __init__(self,
                 img_suffix='_rgb_anon.png',
                 seg_map_suffix='_gt_labelIds.png',
                 ignore_index=255,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ignore_index=ignore_index,
            reduce_zero_label=False,
            **kwargs)


@DATASETS.register_module()
class ISAIDDATASET(BaseSegDataset):
    METAINFO = dict(
        classes=(
            ['others', 'boat', 'storage tank', 'baseball diamond', 'tennis court', 'basketball court',
             'running track field', 'bridge', 'truck or bus', 'car', 'helicopter', 'swimming pool', 'traffic circle',
             'soccer field', 'air plane', 'pier']),
        palette=[[0, 0, 0], [0, 0, 63], [0, 63, 63], [0, 63, 0], [0, 63, 127],
                 [0, 63, 191], [0, 63, 255], [0, 127, 63], [0, 127, 127],
                 [0, 0, 127], [0, 0, 191], [0, 0, 255], [0, 191, 127],
                 [0, 127, 191], [0, 127, 255], [0, 100, 155]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 ignore_index=255,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ignore_index=ignore_index,
            reduce_zero_label=False,
            **kwargs)


@DATASETS.register_module()
class WORLDFLOODSDATASET(BaseSegDataset):
    METAINFO = dict(
        classes=(
            ['land', 'water and flood', 'cloud']),
        palette=[[0, 0, 0], [0, 0, 63], [0, 63, 63]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 ignore_index=255,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ignore_index=ignore_index,
            reduce_zero_label=False,
            **kwargs)


@DATASETS.register_module()
class UAVIDDATASET(BaseSegDataset):
    METAINFO = dict(
        classes=(
            ['others', 'building', 'road', 'tree', 'grass', 'moving car', 'parked car', 'humans']),
        palette=[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7], ])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 ignore_index=255,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ignore_index=ignore_index,
            reduce_zero_label=False,
            **kwargs)


@DATASETS.register_module()
class KVASIR_INSTRUMENTDATASET(BaseSegDataset):
    METAINFO = dict(
        classes=(
            ['others', 'tool']),
        palette=[[0, 0, 0], [1, 1, 1]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 ignore_index=255,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ignore_index=ignore_index,
            reduce_zero_label=False,
            **kwargs)


@DATASETS.register_module()
class CORROSIONDATASET(BaseSegDataset):
    METAINFO = dict(
        classes=(
            ['others',
             'steel with fair corrosion',
             'steel with poor corrosion',
             'steel with severe corrosion', ]),
        palette=[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])

    def __init__(self,
                 img_suffix='.jpeg',
                 seg_map_suffix='.png',
                 ignore_index=255,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ignore_index=ignore_index,
            reduce_zero_label=False,
            **kwargs)


@DATASETS.register_module()
class ZeroWasteDATASET(BaseSegDataset):
    METAINFO = dict(
        classes=(
            ['background or trash',
             'rigid plastic',
             'cardboard',
             'metal',
             'soft plastic', ]),
        palette=[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [3, 3, 4]])

    def __init__(self,
                 img_suffix='.PNG',
                 seg_map_suffix='.PNG',
                 ignore_index=255,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ignore_index=ignore_index,
            reduce_zero_label=False,
            **kwargs)


@DATASETS.register_module()
class DeepCrackDATASET(BaseSegDataset):
    METAINFO = dict(
        classes=(
            ['concrete or asphalt', 'crack']),
        palette=[[0, 0, 0], [1, 1, 1]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 ignore_index=255,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ignore_index=ignore_index,
            reduce_zero_label=False,
            **kwargs)


@DATASETS.register_module()
class PST900DATASET(BaseSegDataset):
    METAINFO = dict(
        classes=(
            ['background',
             'fire extinguisher',
             'backpack',
             'drill',
             'human', ]),
        palette=[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 ignore_index=255,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ignore_index=ignore_index,
            reduce_zero_label=False,
            **kwargs)


@DATASETS.register_module()
class FLOODNETDATASET(BaseSegDataset):
    METAINFO = dict(
        classes=(
            ['background',
             'building-flooded',
             'building-non-flooded',
             'road-flooded',
             'road-non-flooded',
             'water',
             'tree',
             'vehicle',
             'pool',
             'grass', ]),
        palette=[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 ignore_index=255,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ignore_index=ignore_index,
            reduce_zero_label=False,
            **kwargs)


@DATASETS.register_module()
class CryoNuSegDATASET(BaseSegDataset):
    METAINFO = dict(
        classes=(
            ['others', 'nuclei in cells']),
        palette=[[0, 0, 0], [1, 1, 1]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 ignore_index=255,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ignore_index=ignore_index,
            reduce_zero_label=False,
            **kwargs)


@DATASETS.register_module()
class PAXRAYDATASET(BaseSegDataset):
    METAINFO = dict(
        classes=(
            ['others', 'organ']),
        palette=[[0, 0, 0], [1, 1, 1]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 ignore_index=255,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ignore_index=ignore_index,
            reduce_zero_label=False,
            **kwargs)
