from bosk.imports import *
from bosk.__init__ import *
from bosk.utils import make_prediction

def get_patches(image: Image.Image, num_patches: int) -> List[np.ndarray]:
    """Разбивает изображение на num_patches кусков (патчей).
    Args:
        image: Исходное изображение (PIL.Image).
        num_patches: Количество патчей, на которые нужно разбить изображение.
    Return:
        patches: Список патчей (каждый патч — это numpy array).
    """
    image_array = np.array(image)

    height, width, _ = image_array.shape

    # Вычисляем количество патчей по вертикали и горизонтали
    patches_per_side = int(np.sqrt(num_patches))
    if patches_per_side ** 2 != num_patches:
        raise ValueError("num_patches должно быть полным квадратом (например, 4, 9, 16, ...)")

    # Вычисляем размер каждого патча
    patch_height = height // patches_per_side
    patch_width = width // patches_per_side

    # Список для хранения патчей
    patches = []

    # Разбиваем изображение на патчи
    for i in range(patches_per_side):
        for j in range(patches_per_side):
            # Вычисляем координаты начала и конца патча
            start_y = i * patch_height
            end_y = start_y + patch_height
            start_x = j * patch_width
            end_x = start_x + patch_width

            patch = image_array[start_y:end_y, start_x:end_x, :]
            patches.append(patch)

    return patches


def combine_patches(patches: List[np.ndarray]) -> Image.Image:
    """Объединяет патчи в единое изображение
    Args:
        patches: Набор патчей, которые необходимо объединить в единое изображение.
    Return:
        image: Объединенное изображение.
    """
    num_patches = len(patches)
    patch_size = patches[0].shape[0]
    patches_per_side = int(np.sqrt(num_patches))
    if len(patches[0].shape) > 2:
        num_channels = patches[0].shape[2]
    else:
        num_channels=1

    image_size = patches_per_side * patch_size
    image = np.zeros((image_size, image_size, num_channels))
    for i, patch in enumerate(patches):
        row_idx = i // patches_per_side * patch_size
        col_idx = i % patches_per_side * patch_size
        image[row_idx:row_idx + patch_size, col_idx:col_idx + patch_size] = \
            patch.reshape(patch_size, patch_size, num_channels)
    
    image = Image.fromarray(image.reshape(image_size, image_size))
    return image


def get_patches_with_intersection(image: Image.Image, num_patches: int) -> List[np.ndarray]:
    """Разбивает изображение на num_patches кусков (патчей) с пересечением в пловоину размеров patch'а.
    Args:
        image: Исходное изображение (PIL.Image).
        num_patches: Количество патчей, на которые нужно разбить изображение.
    Return:
        patches: Список патчей (каждый патч — это numpy array).
    """
    image_array = np.array(image)

    height, width, _ = image_array.shape

    # Вычисляем количество патчей по вертикали и горизонтали
    patches_per_side = int(np.sqrt(num_patches))
    if patches_per_side ** 2 != num_patches:
        raise ValueError("num_patches должно быть полным квадратом (например, 4, 9, 16, ...)")

    # Вычисляем размер каждого патча
    patch_height = height // patches_per_side
    patch_width = width // patches_per_side

    # Список для хранения патчей
    patches = []

    # Разбиваем изображение на патчи c пересечением в половину размера patch'а
    for i in range(2 * patches_per_side - 1):
        for j in range(2 * patches_per_side - 1):
            # Вычисляем координаты начала и конца патча
            start_y = i * (patch_height // 2)
            end_y = start_y + patch_height
            start_x = j * (patch_width // 2)
            end_x = start_x + patch_width

            patch = image_array[start_y:end_y, start_x:end_x, :]
            patches.append(patch)

    return patches


def combine_patches_with_intersection(
        patches: List[np.ndarray]
) -> Image.Image:
    """Объединяет патчи в единое изображение
    Args:
        patches: Набор патчей, которые необходимо объединить в единое изображение.
    Return:
        image: Объединенное бинаризованное изображение.
    """
    num_patches = len(patches)
    patch_size = patches[0].shape[0]
    patches_per_side = int(np.sqrt(num_patches))
    if len(patches[0].shape) > 2:
        num_channels = patches[0].shape[2]
    else:
        num_channels=1

    image_size = (patches_per_side + 1) // 2  * patch_size
    image = np.zeros((image_size, image_size, num_channels)).astype("int")
    for i, patch in enumerate(patches):
        row_idx = i // patches_per_side * (patch_size // 2)
        col_idx = i % patches_per_side * (patch_size // 2)
        patch = patch.reshape(patch_size, patch_size, num_channels)
        # бинаризуем маску сегментриованного patch'а
        patch[~np.isclose(patch, 0.0)] = 1.0
        image[row_idx:row_idx + patch_size, col_idx:col_idx + patch_size] = np.bitwise_or(
            image[row_idx:row_idx + patch_size, col_idx:col_idx + patch_size], patch
        )
    image = image.astype("float32")
    image = Image.fromarray(image.reshape(image_size, image_size)).convert("L")
    return image


def segment_patch(
    model,
    image,
    model_type,
    transform,
    visualization=False,
    device=DEVICE
):
    """Данная функция визуализирует сегментацию изображения image
    Args:
        model: модель для сегментации
        image: изображения для сегментации
        model_type: аргумент для функции make_prediction, описывающий как
        произвести предсказания в зависимости от типа модели
        transform: трансформация входного изображения image
        device: тип девайса на котором производить вычисления
    Return:
        predicted_mask: сегментированное изображение (тип - np.array())
    """
    image = image.convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Предсказание
    with torch.no_grad():
        output = make_prediction(model, input_tensor, model_type)
        predicted_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # === Визуализация ===
    if visualization:
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

    return np.array(predicted_mask)
