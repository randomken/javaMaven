package springbootAI;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;

import javax.imageio.ImageIO;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.nd4j.enums.ImageResizeMethod;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;


@RestController
@RequestMapping("/faceMatch")
public class FaceMatching {
	private static String localpath = "C:\\Users\\KahKen\\Downloads\\Self Trained Revised_V4\\";
	private static String[] className = {"AIA_6702_nolabel",
							            "AIA_8659_nolabel",
							            "AIA_aia_nolabel",	
							            "AIA_part5_nolabel",
							            "AIA_u33_aia(3)_nolabel",
							            "Aetna_3542_nolabel",
							            "Aetna_Aetna_nolabel",
							            "Affin_E417D_nolabel",
							            "Allianz_8556_nolabel",
							            "Etiqa_u13 (1)_nolabel",
							            "Etiqa_u13 (2)_nolabel",
							            "Foreign_aiasg_nolabel",
							            "GE_0400_nolabel",
							            "GE_2106_nolabel",
							            "GE_4A4A_nolabel",
							            "GE_5393_nolabel",
							            "GE_6130_nolabel",
							            "GE_7FDF_nolabel",
							            "GE_ge1003_nolabel",
							            "GE_ge_nolabel",
							            "GE_policy_page02_nolabel",
							            "HLA_hla_nolabel",
							            "ING_ing_nolabel",
							            "OAC_FB0A_nolabel",
							            "OAC_u26(2)_nolabel",
							            "Prudential_pru7347_pg1pg2_nolabel",
							            "Prudential_prulady_pg1pg2pg3_nolabel",
							            "Prudential_u23_prudential (1)(2)_nolabel",
							            "Prudential_u28_prudential (1)(2)_nolabel",
							            "Prudential_u30_prudential2 (1)(2)(3)_nolabel",
							            "Prudential_u33_prudential (1)(2)(3)_nolabel",
							            "Prudential_u5_prudential (1)(2)_nolabel",
							            "Prudential_u_poldoc_pg1pg2pg3pg4_nolabel",
							            "Prudential_u_pru9000_pt1_nolabel",
							            "Sunlife_sunlife_nolabel",
							            "Unknown_u14_nolabel"
							            };
	private static String[] pageName = {"1","2","3","4"};
	@RequestMapping("/run")
	public void method() throws Exception{
		System.out.println("run");
		System.out.println("get path");
		String fullModel = new File(localpath+"policy_full_model.h5").getPath();
		ClassLoader classloader = Thread.currentThread().getContextClassLoader();
		InputStream inputStream = classloader.getResourceAsStream("model/policy_full_model.h5");
		//InputStream inputStream =//ClassLoader.getResourceAsStream("model/policy_full_model.h5");
		//String modelJson = new File(localpath+"policy_model_config.json").getPath();
		//String modelWeights = new File(localpath+"policy_model_weights.h5").getPath();
		System.out.println("end get path");
		int resizeWidth =159, resizeHeight=225;
		//load img
		BufferedImage myImage = ImageIO.read(classloader.getResourceAsStream("Sample image testing/sample_upload_AIA_6702.jpeg"));
//		BufferedImage bufferedImageResult = new BufferedImage(
//				resizeWidth,
//				resizeHeight,
//		        myImage.getType()
//		);
//		Graphics2D g2d = bufferedImageResult.createGraphics();
//		g2d.drawImage(
//				myImage, 
//		        0, 
//		        0, 
//		        resizeWidth, 
//		        resizeHeight, 
//		        null
//		);
//		g2d.dispose();
//		String imagePathToWrite = localpath+"image.png";
//		String formatName = imagePathToWrite.substring(
//		        imagePathToWrite.lastIndexOf(".") + 1
//		);
//		ImageIO.write(
//		        bufferedImageResult, 
//		        formatName, 
//		        new File(imagePathToWrite)
//		);
		NativeImageLoader loader1 = new NativeImageLoader(resizeHeight, resizeWidth, 3);
		NativeImageLoader loader2 = new NativeImageLoader(myImage.getHeight(), myImage.getWidth(), 3);
//		System.out.println(loader.asMatrix(bufferedImageResult).shapeInfoToString());
		System.out.println(loader1.asMatrix(myImage).shapeInfoToString());
		INDArray fiximg = loader1.asMatrix(myImage).reshape(1,resizeHeight,resizeWidth,3);
		INDArray img = loader2.asMatrix(myImage).reshape(1,myImage.getHeight(),myImage.getWidth(),3);
		System.out.println("img"+img.shapeInfoToString());
		INDArray heightWeightArray = new Nd4j().create(new int[] {225,159},new long[] {2},DataType.INT);
		System.out.println("heightWeightArray"+heightWeightArray.dataType());
		System.out.println("heightWeightArray"+heightWeightArray.toString());
		System.out.println("heightWeightArray"+heightWeightArray.shapeInfoToString());
		INDArray expandImg = new Nd4j().image().imageResize(img,heightWeightArray, ImageResizeMethod.ResizeNearest);
		//img =new NDImage().imageResize(img,fiximg, null);
		System.out.println("reshape array:"+fiximg.shapeInfoToString());
		System.out.println("expandImg array:"+expandImg.shapeInfoToString());
		System.out.println(loader1.asMatrix(myImage).shapeInfoToString());
		//System.out.println("expandImg :" );
		//System.out.println(expandImg.toString());
//		System.out.println(loader.asMatrix(bufferedImageResult).toString());
		//reshape img
		ImagePreProcessingScaler imageScaler = new ImagePreProcessingScaler();
		//ComputationGraphConfiguration modelConfig = KerasModelImport.importKerasModelConfiguration(modelJson);
		
		
		//MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(fullModel);
		//model1.output(img);
		//ComputationGraph model = KerasModelImport.importKerasModelAndWeights(modelJson, modelWeights);
		ComputationGraph model = KerasModelImport.importKerasModelAndWeights(inputStream, false);
		//MultiLayerNetwork net2 = MultiLayerNetwork.load(new File(localpath+"Model\\Cosine\\20220715-075558\\saved_model.pb"), true);
        //predict
		
        INDArray[] output = model.output(expandImg);
        System.out.println("predict done ");
        for(INDArray out:output)
        System.out.println(out.toString());
		
        String className =  this.className[output[0].argMax().getInt(0)];
        String pageName =  this.pageName[output[1].argMax().getInt(0)];
        System.out.println("className:"+className+" pageName:"+pageName);
//        ImageIO.write(
//        		imageFromINDArray(output[0]), 
//		        formatName, 
//		        new File(imagePathToWrite)
//		);
	}
	
	private BufferedImage imageFromINDArray(INDArray array) {
	    long[] shape = array.shape();

	    long height = shape[2];
	    long width = shape[3];
	    BufferedImage image = new BufferedImage((int)width, (int)height, BufferedImage.TYPE_INT_RGB);
	    for (int x = 0; x < width; x++) {
	        for (int y = 0; y < height; y++) {
	            int red = array.getInt(0, 2, y, x);
	            int green = array.getInt(0, 1, y, x);
	            int blue = array.getInt(0, 0, y, x);

	            //handle out of bounds pixel values
	            red = Math.min(red, 255);
	            green = Math.min(green, 255);
	            blue = Math.min(blue, 255);

	            red = Math.max(red, 0);
	            green = Math.max(green, 0);
	            blue = Math.max(blue, 0);
	            image.setRGB(x, y, new Color(red, green, blue).getRGB());
	        }
	    }
	    return image;
	}
}
