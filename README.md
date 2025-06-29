
Sys.setenv(LANG = "en")

# Pakete laden
library(tidyverse)
library(caret)
library(pROC)
library(PRROC)
library(dbscan)
library(isotree)
library(smotefamily)

# Daten einlesen
df <- read.csv("C:/Users/Administrator/Downloads/archive (3)/creditcard.csv")
df$Class <- factor(df$Class, levels = c(0, 1))

# Spalten extrahieren
features <- df[, grepl("^V\\d+$", names(df))]
df$Amount <- scale(df$Amount)

# Trainings-/Testaufteilung
set.seed(123)
split <- createDataPartition(df$Class, p = 0.8, list = FALSE)
train_data <- df[split, ]
test_data <- df[-split, ]

# z-Score Analyse (auf Betrag)
train_data$z_score <- abs(train_data$Amount)
train_data$z_flag <- ifelse(train_data$z_score > 3, 1, 0)

# Mahalanobis-Distanz
train_feat <- train_data[, grepl("^V\\d+$", names(train_data))]
mah_dist <- mahalanobis(train_feat, colMeans(train_feat), cov(train_feat))
mah_thresh <- quantile(mah_dist, 0.995)
train_data$mah_flag <- ifelse(mah_dist > mah_thresh, 1, 0)

# Isolation Forest
model_if <- isolation.forest(train_feat, ntrees = 100)
test_feat <- test_data[, grepl("^V\\d+$", names(test_data))]
iso_scores <- predict(model_if, test_feat)
iso_pred <- ifelse(iso_scores > quantile(iso_scores, 0.995), 1, 0)
iso_pred <- factor(iso_pred, levels = c(0, 1))

# Konfusionsmatrix
conf <- confusionMatrix(iso_pred, test_data$Class, positive = "1")
conf_df <- as.data.frame(conf$table)
names(conf_df) <- c("Vorhersage", "Tatsächlich", "Freq")

if (!dir.exists("bilder")) dir.create("bilder")
pdf("bilder/konfusionsmatrix.pdf")
ggplot(conf_df, aes(x = Tatsächlich, y = Vorhersage, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), size = 5) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  theme_minimal() +
  labs(title = "Konfusionsmatrix – Isolation Forest")
dev.off()

# ROC-Kurve
roc_obj <- roc(test_data$Class, iso_scores)
pdf("bilder/roc_kurve.pdf")
plot(roc_obj, main = "ROC-Kurve – Isolation Forest", col = "darkblue", lwd = 2)
dev.off()

# Precision-Recall
pr <- pr.curve(scores.class0 = iso_scores[test_data$Class == 1],
               scores.class1 = iso_scores[test_data$Class == 0], curve = TRUE)
pdf("bilder/precision_recall.pdf")
plot(pr, main = "Precision-Recall-Kurve – Isolation Forest", col = "darkred", lwd = 2)
dev.off()

# PCA für alle V-Variablen
pca <- prcomp(df[, grepl("^V\\d+$", names(df))], scale. = TRUE)
pca_df <- as.data.frame(pca$x[, 1:2])
pca_df$Class <- df$Class

pdf("bilder/pca_plot.pdf")
ggplot(pca_df, aes(x = PC1, y = PC2, color = Class)) +
  geom_point(alpha = 0.4) +
  theme_minimal() +
  labs(title = "PCA-Visualisierung der Transaktionen")
dev.off()

# SMOTE
set.seed(123)

smote_result <- SMOTE(X = train_data[, -which(names(train_data) == "Class")],
                      target = train_data$Class,
                      K = 5)

colnames(smote_result$data)

train_balanced <- smote_result$data
train_balanced$Class <- as.factor(train_balanced$class)
train_balanced$class <- NULL  

table(train_balanced$Class)


# Modelle mit/ohne SMOTE
model_orig <- isolation.forest(train_data[, -which(names(train_data) == "Class")])
model_smote <- isolation.forest(train_balanced[, -which(names(train_balanced) == "Class")])

# Scores berechnen
scores_orig <- predict(model_orig, newdata = test_data[, -which(names(test_data) == "Class")])
scores_smote <- predict(model_smote, newdata = test_data[, -which(names(test_data) == "Class")])
threshold <- 0.65
pred_orig <- ifelse(scores_orig > threshold, 1, 0)
pred_smote <- ifelse(scores_smote > threshold, 1, 0)

# Evaluation
confusionMatrix(as.factor(pred_orig), as.factor(test_data$Class), positive = "1")
confusionMatrix(as.factor(pred_smote), as.factor(test_data$Class), positive = "1")

# AUPRC
pr_orig <- pr.curve(scores.class0 = scores_orig[test_data$Class == 1],
                    scores.class1 = scores_orig[test_data$Class == 0],
                    curve = TRUE)
pr_smote <- pr.curve(scores.class0 = scores_smote[test_data$Class == 1],
                     scores.class1 = scores_smote[test_data$Class == 0],
                     curve = TRUE)

# Baseline
baseline <- mean(test_data$Class == 1)
cat("Baseline (Anteil positiver Klasse):", round(baseline, 5))

# Zusätzliche PCA für Time & Amount
scaled_vars <- scale(train_data[, c("Time", "Amount")])
pca_extra <- prcomp(scaled_vars, center = TRUE, scale. = TRUE)

