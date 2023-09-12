from imageai.Classification.Custom import CustomImageClassification

prediction = CustomImageClassification()
prediction.setModelTypeAsMobileNetV2()
prediction.setModelPath("./jenis-makanan-indo/models/mobilenet_v2-jenis-makanan-indo-test_acc_1.00000_epoch-13.pt")
prediction.setJsonPath("./jenis-makanan-indo/models/jenis-makanan-indo_model_classes.json")
prediction.loadModel()

predictions, probabilities = prediction.classifyImage("rendang2.jpg", result_count=2)

for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)