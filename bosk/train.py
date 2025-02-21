from bosk.imports import *
from bosk.utils import make_prediction


device = "cuda" if torch.cuda.is_available() else "cpu"


def train(model, train_dataloader, valid_dataloader, criterion, lr, epochs, model_name, num_classes=5, verbose=False, is_scheduler=False):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    iou_score = JaccardIndex(task="multiclass" if num_classes > 2 else "binary", num_classes=num_classes).to(device)
    dice_score = Dice(num_classes=num_classes, threshold=0.5, zero_division=1e-8).to(device)
    
    if is_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Получаем первый батч валидации (фиксируем его)
    val_data, val_true_mask = next(iter(valid_dataloader))
    val_data, val_true_mask = val_data.to(device), val_true_mask.to(device)

    best_model = None
    best_val_loss = float("inf")  # Инициализируем наихудшее значение лосса
    best_val_iou_score = 0
    best_val_dice_score = 0

    loss_history       = {"train": [], "valid": []}
    iou_score_history  = {"train": [], "valid": []}
    dice_score_history = {"train": [], "valid": []}

    # Начальная проверка
    model.eval()
    with torch.no_grad():
        val_pred = make_prediction(model, val_data, model_name)
        predicted_classes = torch.argmax(val_pred, dim=1)       # Теперь [B, H, W]
        initial_val_loss = criterion(val_pred, val_true_mask).item()
        initial_val_iou_score = iou_score(predicted_classes, val_true_mask).detach()
        initial_val_dice_score = dice_score(predicted_classes, val_true_mask).detach()

    print(f"Initial validation loss: {initial_val_loss:.4f}")
    print(f"Initial validation IoU score: {initial_val_iou_score:.4f}")
    print(f"Initial validation Dice score: {initial_val_dice_score:.4f}")

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Обучение
        model.train()
        total_loss = 0
        total_iou_score, total_dice_score = 0, 0
        
        for data, true_mask in tqdm(train_dataloader, desc="Training", leave=False):
            data, true_mask = data.to(device), true_mask.to(device)
            optimizer.zero_grad()

            prediction = make_prediction(model, data, model_name)
            loss = criterion(prediction, true_mask)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_iou_score += iou_score(prediction, true_mask).item()
            total_dice_score += dice_score(prediction, true_mask).item()

        train_loss = total_loss / len(train_dataloader)
        train_iou_score = total_iou_score / len(train_dataloader)
        train_dice_score = total_dice_score / len(train_dataloader)

        # Шаг learning rate scheduler'а
        if is_scheduler:
            scheduler.step()

        # Оценка на валидации
        torch.cuda.empty_cache()  # Очищаем кеш перед валидацией
        model.eval()
        with torch.no_grad():
            val_pred = make_prediction(model, val_data, model_name)
            predicted_classes = torch.argmax(val_pred, dim=1)  # Теперь [B, H, W]
            val_loss = criterion(val_pred, val_true_mask).item()
            val_iou_score = iou_score(predicted_classes, val_true_mask).detach()
            val_dice_score = dice_score(predicted_classes, val_true_mask).detach()

        # Сохраняем лучшую модель
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_iou_score = val_iou_score
            best_val_dice_score = val_dice_score
            best_model = deepcopy(model)  # Глубокая копия, а не ссылка

        # Логирование
        loss_history["train"].append(train_loss)
        loss_history["valid"].append(val_loss)
        iou_score_history["train"].append(train_iou_score)
        iou_score_history["valid"].append(val_iou_score)
        dice_score_history["train"].append(train_dice_score)
        dice_score_history["valid"].append(val_dice_score)

        if verbose:
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Train IoU score: {train_iou_score:.4f}, Val IoU score: {val_iou_score:.4f}")
            print(f"Train Dice score: {train_dice_score:.4f}, Val Dice score: {val_dice_score:.4f}")

    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Validation IoU score: {best_val_iou_score:.4f}")
    print(f"Best Validation Dice score: {best_val_dice_score:.4f}")

    return best_model, loss_history, iou_score_history, dice_score_history
