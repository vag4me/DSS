import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Attribute;

public class WekaPrediction {
    public static void main(String[] args) {
        try {
            // Step 1: Load the pre-trained model
            Classifier model = (Classifier) SerializationHelper.read("C:/Users/user/Desktop/Ionio/python/DSS/vote.model");

            // Step 2: Load the new data
            Instances newData = new Instances(new java.io.FileReader("C:/Users/user/Desktop/Ionio/python/DSS/vote.arff"));
            newData.setClassIndex(newData.numAttributes() - 1);

            // Step 3: Make predictions using the model
            for (int i = 0; i < newData.numInstances(); i++) {
                Instance instance = newData.instance(i);
                double prediction = model.classifyInstance(instance);
                
                // Step 4: Output the predictions
                Attribute classAttribute = newData.classAttribute();
                String className = classAttribute.value((int) prediction);
                System.out.println("Predicted class for instance " + (i+1) + ": " + className);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
