from imageai.Classification.Custom import ClassificationModelTrainer

model_trainer = ClassificationModelTrainer()
model_trainer.setModelTypeAsMobileNetV2()
model_trainer.setDataDirectory(r"D:/Developments/python/image-recognition/jenis-makanan-indo")
model_trainer.trainModel(num_experiments=100, batch_size=32)