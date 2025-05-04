
# Pakete laden
library(tidyverse)
library(caret)
library(pROC)
library(PRROC)
library(dbscan)
library(isotree)

# Daten einlesen
df <- read.csv("creditcard.csv")
df$Class <- factor(df$Class, levels = c(0, 1))

# Spalten extrahieren
features <- df[, grepl("^V\d+$", names(df))]
df$Amount <- scale(df$Amount)

# Trainings-/Testaufteilung
set.seed(123)
split <- createDataPartition(df$Class, p = 0.8, list = FALSE)
train <- df[split, ]
test <- df[-split, ]
train_feat <- train[, grepl("^V\d+$", names(train))]
test_feat <- test[, grepl("^V\d+$", names(test))]
train_class <- train$Class
test_class <- test$Class

# z-Score Analyse (auf Betrag)
train$z_score <- abs(train$Amount)
train$z_flag <- ifelse(train$z_score > 3, 1, 0)

# Mahalanobis-Distanz
mah_dist <- mahalanobis(train_feat, colMeans(train_feat), cov(train_feat))
mah_thresh <- quantile(mah_dist, 0.995)
train$mah_flag <- ifelse(mah_dist > mah_thresh, 1, 0)

# Isolation Forest
model_if <- isolation.forest(train_feat, ntrees = 100)
iso_scores <- predict(model_if, test_feat)
iso_pred <- ifelse(iso_scores > quantile(iso_scores, 0.995), 1, 0)
iso_pred <- factor(iso_pred, levels = c(0, 1))

# Konfusionsmatrix
conf <- confusionMatrix(iso_pred, test_class, positive = "1")
conf_df <- as.data.frame(conf$table)
names(conf_df) <- c("Vorhersage", "Tatsächlich", "Freq")

pdf("bilder/konfusionsmatrix.pdf")
ggplot(conf_df, aes(x = Tatsächlich, y = Vorhersage, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), size = 5) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  theme_minimal() +
  labs(title = "Konfusionsmatrix – Isolation Forest")
dev.off()

# ROC
roc_obj <- roc(test_class, iso_scores)
pdf("bilder/roc_kurve.pdf")
plot(roc_obj, main = "ROC-Kurve – Isolation Forest", col = "darkblue", lwd = 2)
dev.off()

# Precision-Recall
pr <- pr.curve(scores.class0 = iso_scores[test_class == 1],
               scores.class1 = iso_scores[test_class == 0], curve = TRUE)
pdf("bilder/precision_recall.pdf")
plot(pr, main = "Precision-Recall-Kurve – Isolation Forest", col = "darkred", lwd = 2)
dev.off()

# PCA
pca <- prcomp(df[, grepl("^V\d+$", names(df))], scale. = TRUE)
pca_df <- as.data.frame(pca$x[, 1:2])
pca_df$Class <- df$Class

pdf("bilder/pca_plot.pdf")
ggplot(pca_df, aes(x = PC1, y = PC2, color = Class)) +
  geom_point(alpha = 0.4) +
  theme_minimal() +
  labs(title = "PCA-Visualisierung der Transaktionen")
dev.off()
