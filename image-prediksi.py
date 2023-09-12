from imageai.Classification import ImageClassification
prediction = ImageClassification()

prediction.setModelTypeAsMobileNetV2()
prediction.setModelPath("./mobilenet_v2-b0353104.pth")
prediction.loadModel()
predictions, probabilities = prediction.classifyImage("pel.jpg", result_count=10)
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)