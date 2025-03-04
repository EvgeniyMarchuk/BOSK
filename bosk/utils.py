from bosk import *
from bosk.imports import *


def get_unique_classes(mask_path):
    """
    Опеределяем значения уникальных классов в маске, путь до которой передается в функцию
    """
    mask = Image.open(mask_path).convert("L")  # Градации серого
    mask_array = np.array(mask)

    # Определяем уникальные классы
    unique_classes = np.unique(mask_array)
    print("Уникальные классы в маске:", unique_classes)
    print("Количество классов:", len(unique_classes))


def remove_identifier_files(base_path):
    """
    Удаляет файлы, заканчивающиеся на 'Identifier', в images/{20, 50, 100} и masks/{20, 50, 100}.
    """
    folders = [
        "images/20",
        "images/50",
        "images/100",
        "masks/20",
        "masks/50",
        "masks/100",
    ]

    for folder in folders:
        full_path = os.path.join(base_path, folder)
        if os.path.exists(full_path):
            for file in os.listdir(full_path):
                if file.endswith("Identifier"):
                    os.remove(os.path.join(full_path, file))
                    print(f"Удален: {file}")


def split_dataset(base_path, train_doze=0.7, test_doze=0.2, valid_doze=0.1):
    """
    Разбивает изображения и маски в соотношении 70/20/10 и перемещает в train, test, valid.
    """
    random.seed(42)  # Фиксируем случайность

    sets = {"train": train_doze, "test": test_doze, "valid": valid_doze}
    categories = ["20", "50", "100"]

    for category in categories:
        image_files = sorted(glob(os.path.join(base_path, f"images/{category}/*.png")))
        mask_files = sorted(glob(os.path.join(base_path, f"masks/{category}/*.png")))

        if len(image_files) != len(mask_files):
            raise ValueError(
                f"Несовпадение числа изображений и масок в категории {category}"
            )

        dataset_size = len(image_files)
        indices = list(range(dataset_size))
        random.shuffle(indices)

        split_points = {
            "train": int(sets["train"] * dataset_size),
            "test": int(sets["train"] * dataset_size)
            + int(sets["test"] * dataset_size),
        }

        partitions = {
            "train": indices[: split_points["train"]],
            "test": indices[split_points["train"] : split_points["test"]],
            "valid": indices[split_points["test"] :],
        }

        for set_name, idx_list in partitions.items():
            for idx in idx_list:
                shutil.move(
                    image_files[idx], os.path.join(base_path, f"{set_name}/images")
                )
                shutil.move(
                    mask_files[idx], os.path.join(base_path, f"{set_name}/masks")
                )

    print("Разделение завершено.")


def preprocess_mask(mask):
    """Функция предобработки маски: сохраняем только интересующиеся классы"""
    mask_array = np.array(mask)

    # Создаем пустую маску
    new_mask = np.zeros_like(mask_array, dtype=np.uint8)

    # Назначаем классы
    new_mask[mask_array == CLASS_IDS["road"]] = 1  # Дорога  -> класс 1
    new_mask[mask_array == CLASS_IDS["lake"]] = 2  # Озеро   -> класс 2
    new_mask[mask_array == CLASS_IDS["bridge"]] = 3  # Мост    -> класс 3
    new_mask[mask_array == CLASS_IDS["tree"]] = 4  # Деревья -> класс 4
    new_mask[mask_array == CLASS_IDS["background"]] = 0  # Фон     -> класс 0

    return torch.tensor(new_mask, dtype=torch.long)


class VALID_Dataset(Dataset):
    """Класс датасета протяженных объектов"""
    def __init__(
        self, root_dir, model_type, transforms=None, processor=None, image_size=512
    ):
        self.root_dir = root_dir
        self.image_paths = []
        self.mask_paths = []
        self.processor = processor
        self.transform = transforms
        self.image_size = image_size
        self.model_type = model_type

        img_dir = os.path.join(root_dir, "images")
        mask_dir = os.path.join(root_dir, "masks")

        images_list = sorted(os.listdir(img_dir))
        masks_list = sorted(os.listdir(mask_dir))
        for i, img_name in enumerate(images_list):
            self.image_paths.append(os.path.join(img_dir, img_name))
            self.mask_paths.append(os.path.join(mask_dir, masks_list[i]))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")  # Grayscale

        if self.model_type == "segformer":
            encoding = self.processor(image, return_tensors="pt")
            image = encoding["pixel_values"].squeeze(0)  # [3, H, W]
            mask = mask.resize(
                (self.image_size, self.image_size), resample=Image.NEAREST
            )
        else:
            image = self.transform(image)
            mask = mask.resize(
                (self.image_size, self.image_size), resample=Image.NEAREST
            )
        mask = preprocess_mask(mask)
        mask = torch.tensor(np.array(mask), dtype=torch.long)

        return image, mask


def make_prediction(model, data, model_type):
    """В зависимости от типа модели делаем предсказания моделью для переданных данных"""
    pred = None
    if model_type == "deeplab":
        pred = model(data)["out"]
    if model_type == "unet_mobile":
        pred = model(data)
    if model_type == "segformer":
        pred = model(pixel_values=data).logits
        pred = F.interpolate(
            pred, size=(512, 512), mode="bilinear", align_corners=False
        )
    return pred


def load_model(
    model_path,
    model_type,
    num_classes=NUM_CLASSES,
    device=DEVICE,
):
    model_config = {
        "deeplab": {
            "constructor": deeplabv3_mobilenet_v3_large,
            "modify_layer": lambda model: setattr(
                model.classifier, "4", torch.nn.Conv2d(256, num_classes, kernel_size=1)
            ),
        },
        "segformer": {
            "constructor": SegformerForSemanticSegmentation.from_pretrained,
            "pretrained": "nvidia/segformer-b0-finetuned-ade-512-512",
            "modify_layer": lambda model: setattr(
                model.decode_head,
                "classifier",
                torch.nn.Conv2d(256, num_classes, kernel_size=1),
            ),
            "kwargs": {"num_labels": num_classes, "ignore_mismatched_sizes": True},
        },
        "unet_mobile": {
            "constructor": smp.Unet,
            "kwargs": {
                "encoder_name": "mobilenet_v2",
                "encoder_weights": "imagenet",
                "classes": num_classes,
            },
        },
        "unet_efficient": {
            "constructor": smp.Unet,
            "kwargs": {
                "encoder_name": "efficientnet-b4",
                "encoder_weights": "imagenet",
                "classes": num_classes,
            },
        },
    }

    config = model_config.get(model_type)
    if not config:
        raise ValueError(f"Unknown model type: {model_type}")

    if "pretrained" in config:
        model = config["constructor"](config["pretrained"], **config.get("kwargs", {}))
    else:
        model = config["constructor"](**config.get("kwargs", {}))

    if "modify_layer" in config:
        config["modify_layer"](model)

    # Загружаем веса
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    return model


def load_model_for_large_image(model_path, num_classes=2, device="cuda"):
    # Загружаем предобученную модель
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    ).to(device)

    # Заменяем последний классифицирующий слой
    model.decode_head.classifier = torch.nn.Conv2d(256, 1, kernel_size=1)
    torch.nn.init.xavier_uniform_(model.decode_head.classifier.weight)
    torch.nn.init.zeros_(model.decode_head.classifier.bias)

    # Загружаем веса из файла
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()  # Переводим в режим оценки
    return model


def visualize_losses_and_scores(losses, iou_scores, dice_scores, recall_scores=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    epochs = np.arange(len(losses["train"]))

    # Лоссы
    axes[0].plot(epochs, losses["train"], c="r", label="Train")
    axes[0].scatter(epochs, losses["train"], c="r")
    axes[0].plot(epochs, losses["valid"], c="b", label="Validation")
    axes[0].scatter(epochs, losses["valid"], c="b")
    axes[0].set_title("Train and validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid()

    # IoU Score (переводим тензоры в CPU и numpy)
    train_scores = [
        s.cpu().item() if isinstance(s, torch.Tensor) else s
        for s in iou_scores["train"]
    ]
    valid_scores = [
        s.cpu().item() if isinstance(s, torch.Tensor) else s
        for s in iou_scores["valid"]
    ]

    axes[1].plot(epochs, train_scores, c="r", label="Train")
    axes[1].scatter(epochs, train_scores, c="r")
    axes[1].plot(epochs, valid_scores, c="b", label="Validation")
    axes[1].scatter(epochs, valid_scores, c="b")
    axes[1].set_title("Train and validation IoU score")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("IoU score")
    axes[1].legend()
    axes[1].grid()

    train_scores = [
        s.cpu().item() if isinstance(s, torch.Tensor) else s
        for s in dice_scores["train"]
    ]
    valid_scores = [
        s.cpu().item() if isinstance(s, torch.Tensor) else s
        for s in dice_scores["valid"]
    ]

    axes[2].plot(epochs, train_scores, c="r", label="Train")
    axes[2].scatter(epochs, train_scores, c="r")
    axes[2].plot(epochs, valid_scores, c="b", label="Validation")
    axes[2].scatter(epochs, valid_scores, c="b")
    axes[2].set_title("Train and validation Dice score")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Dice score")
    axes[2].legend()
    axes[2].grid()

    plt.show()


def visualize_results(data, prediction, ground_truth):
    _, axes = plt.subplots(3, len(data), figsize=(12, 12))
    all_data = {
        0: ("Actual Image", data),
        1: ("Prediction", prediction),
        2: ("Ground truth", ground_truth),
    }

    for i in range(3 * len(data)):
        row = i // len(data)
        col = i % len(data)

        axes[row, col].imshow(all_data[row][1][col].reshape(512, 512))
        axes[row, col].set_title(all_data[row][0])


def testing_model(
    model,
    dataloader,
    model_type,
    num_images=5,
    num_classes=NUM_CLASSES,
    visualization=False,
):
    iou_score = JaccardIndex(task="multiclass", num_classes=num_classes).to(DEVICE)
    dice_score = Dice(num_classes=num_classes, threshold=0.5, zero_division=1e-8).to(
        DEVICE
    )
    with torch.no_grad():
        model.eval()
        loss = 0
        resulted_iou_score = 0
        resulted_dice_score = 0
        criterion = nn.CrossEntropyLoss()
        for data, true_mask in dataloader:
            data, true_mask = data.to(DEVICE), true_mask.to(DEVICE)
            logits = make_prediction(model, data, model_type)
            loss += criterion(logits, true_mask)
            predicted_classes = torch.argmax(logits, dim=1)
            resulted_iou_score += iou_score(predicted_classes, true_mask).detach()
            resulted_dice_score += dice_score(predicted_classes, true_mask).detach()
        resulted_loss = loss / len(dataloader)
        resulted_iou_score /= len(dataloader)
        resulted_dice_score /= len(dataloader)
        print("Loss = ", resulted_loss.cpu())
        print("IoU score = ", resulted_iou_score.cpu())
        print("Dice score = ", resulted_dice_score.cpu())

        if visualization:
            data, true_mask = next(iter(dataloader))
            data, true_mask = data.to(DEVICE), true_mask.to(DEVICE)
            logits = make_prediction(model, data, model_type)
            predicted_classes = torch.argmax(logits, dim=1)
            visualize_results(
                data[:num_images, 0].cpu(),
                predicted_classes[:num_images].cpu(),
                true_mask[:num_images].cpu(),
            )

    return data[:num_images, 0], predicted_classes[:num_images], true_mask[:num_images]


def predict_for_one_image(
    model,
    image_path,
    model_type,
    transform,
    device=DEVICE,
):
    # Загружаем изображение
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Предсказание
    with torch.no_grad():
        output = make_prediction(model, input_tensor, model_type)
        predicted_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # === Визуализация ===
    plt.figure(figsize=(12, 5))

    # Оригинальное изображение
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.axis("off")
    plt.title("Original Image")

    # Предсказанная маска
    plt.subplot(1, 3, 2)
    plt.imshow(predicted_mask, cmap="jet", alpha=0.7)
    plt.axis("off")
    plt.title("Predicted Segmentation")

    # Наложение маски на изображение
    plt.subplot(1, 3, 3)
    plt.imshow(image)
    plt.imshow(predicted_mask, cmap="jet", alpha=0.5)
    plt.axis("off")
    plt.title("Overlay")

    plt.show()

__all__ = [
    "get_unique_classes",
    "remove_identifier_files",
    "split_dataset",
    "preprocess_mask",
    "VALID_Dataset",
    "make_prediction",
    "load_model",
    "load_model_for_large_image",
    "visualize_losses_and_scores",
    "visualize_results",
    "testing_model",
    "predict_for_one_image",
]
