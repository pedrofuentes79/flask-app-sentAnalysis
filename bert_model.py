from transformers import BertForSequenceClassification, BertTokenizer
import torch
import pytorch_lightning as pl

class SentimentClassifier(pl.LightningModule):
    def __init__(self, learning_rate=2e-5):
        super(SentimentClassifier, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
        self.learning_rate = learning_rate
        self.loss_fn = torch.nn.BCELoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits

        labels = labels.float() if labels is not None else None  # Convert labels to Float data type

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        # get the probs and then flatten them to be fed into the loss function.
        # otherwise it will cause shape conflicts, since the labels are flattened.
        probs = self(input_ids, attention_mask).view(-1)
        loss = torch.nn.BCELoss()(probs, labels.float())

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch

        probs = self(input_ids, attention_mask).view(-1)
        loss = torch.nn.BCELoss()(probs, labels.float())

        # Convert probabilities to binary predictions (0 or 1) based on a threshold
        preds = (probs >= 0.5).long()

        correct = (preds == labels).sum().item()
        total = labels.size(0)

        self.log('val_loss', loss)
        self.log('val_accuracy', correct / total, prog_bar=True)

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        probs = self(input_ids, attention_mask)

        # Convert probabilities to binary predictions (0 or 1) based on a threshold
        preds = (probs >= 0.5).long().view(-1)

        # accuracy
        correct = (preds == labels).sum().item()
        total = labels.size(0)
        self.log('test_accuracy', correct / total)


        TP = ((preds == 1) & (labels == 1)).sum()
        TN = ((preds == 0) & (labels == 0)).sum()
        FN = ((preds == 0) & (labels == 1)).sum()
        FP = ((preds == 1) & (labels == 0)).sum()

        # all positive and negative predictions, not the actual labels
        AP = (preds == 1).sum()
        AN = (preds == 0).sum()

        # precision for positive and negative sentiment
        precision_1 = TP / AP if AP != 0 else 0
        precision_0 = TN / AN if AN != 0 else 0

        # recall for positive and negative sentiment
        recall_1 = TP / (TP + FN) if (TP + FN != 0) else 0
        recall_0 = TN / (TN + FP) if (TN + FP != 0) else 0

        # f1 for positive and negative sentiment
        f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1) if (precision_1 + recall_1 != 0) else 0
        f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0) if (precision_0 + recall_0 != 0) else 0

        # log the metrics
        self.log('test_precision_1', precision_1)
        self.log('test_precision_0', precision_0)
        self.log('test_recall_1', recall_1)
        self.log('test_recall_0', recall_0)
        self.log('test_f1_1', f1_1)
        self.log('test_f1_0', f1_0)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def predict_text(input_text):
        if not isinstance(input_text, str):
            raise TypeError("Input text must be a string!")
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = SentimentClassifier.load_from_checkpoint('BERT.ckpt')
        model.eval()

        encoded_text = tokenizer.encode_plus(input_text, return_attention_mask="True", return_tensors="pt")
        input_ids = encoded_text['input_ids']
        attention_mask = encoded_text['attention_mask']

        output = model(input_ids, attention_mask)
        prediction = torch.sigmoid(output["logits"])
        return prediction.item()