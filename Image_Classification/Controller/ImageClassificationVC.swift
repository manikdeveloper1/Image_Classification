//
//  ImageClassificationVC.swift
//  Image_Classification
//
//  Created by Manik Goel on 07/04/25.
//

import UIKit
import TensorFlowLite
import PhotosUI

class ImageClassificationVC: UIViewController, PHPickerViewControllerDelegate {
    
    // MARK: - Variable
    private var interpreter: Interpreter?
    
    // MARK: - @IBOutlet
    @IBOutlet weak private var lblResult: UILabel!
    @IBOutlet weak private var imgObject: UIImageView!
    @IBOutlet weak private var viewUploadImage: UIView!
    @IBOutlet weak private var viewPredict: UIView!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setUI()
    }
    
    // MARK: - Private Functions
    private func setUI() {
        imgObject.layer.cornerRadius = 16  // or any value you want
        imgObject.layer.borderWidth = 2
        imgObject.layer.borderColor = UIColor.lightGray.cgColor
        imgObject.clipsToBounds = true
        viewPredict.clipsToBounds = true
        viewUploadImage.clipsToBounds = true
        viewUploadImage.layer.cornerRadius = 24
        viewPredict.layer.cornerRadius = 24
        interpreter = loadModel()
        lblResult.isHidden = true
        // hii
    }
    
    // To load model from .tflite file
    private func loadModel() -> Interpreter? {
        guard let modelPath = Bundle.main.path(forResource: "Cat_Dog_Classification", ofType: "tflite") else {
            print("Failed to load model.")
            return nil
        }
        do {
            let interpreter = try Interpreter(modelPath: modelPath)
            try interpreter.allocateTensors()
            return interpreter
        } catch {
            print("TensorFlow Lite model loading error: \(error)")
            return nil
        }
    }
    
    // To get image from the gallery
    private func showPicker() {
        var config = PHPickerConfiguration()
        config.filter = .images // ✅ Pick only images
        config.selectionLimit = 1
        let picker = PHPickerViewController(configuration: config)
        picker.delegate = self
        present(picker, animated: true)
    }
    
    func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
        guard let itemProvider = results.first?.itemProvider else { return }
        if itemProvider.canLoadObject(ofClass: UIImage.self) {
            itemProvider.loadObject(ofClass: UIImage.self) { [weak self] image, error in
                if let error = error {
                    print("❌ Error loading image: \(error.localizedDescription)")
                    return
                }
                guard let image = image as? UIImage else { return }
                // ✅ Use the image (e.g., save, display, etc.)
                DispatchQueue.main.async {
                    print("✅ Image loaded: \(image)")
                    self?.imgObject.image = image
                    self?.lblResult.text = ""
                }
            }
        }
        picker.dismiss(animated: true)
    }
    
    // To preprocess the image before feeding it into the TensorFlow Lite model (resize, normalize, and convert to the correct format).
    private func preprocessImage(_ image: UIImage, size: CGSize = CGSize(width: 265, height: 265)) -> Data? {
        guard let inputTensor = try? interpreter?.input(at: 0) else { return nil }
        let shape = inputTensor.shape.dimensions
        let height = shape[1]
        let width = shape[2]
        return image.resize(to: CGSize(width: width, height: height))?.normalizedRGBData()
    }
    
    // MARK: - @IBAction
    @IBAction func onTapPredict(_ sender: UIButton) {
        guard let image = imgObject.image, let inputData = preprocessImage(image), let interpreter = interpreter else {
            print("Missing image or interpreter")
            return
        }
        do {
            try interpreter.copy(inputData, toInputAt: 0)
            try interpreter.invoke()
            let outputTensor = try interpreter.output(at: 0)
            let results = outputTensor.data.withUnsafeBytes { buffer -> [Float] in
                let floatBuffer = buffer.bindMemory(to: Float.self)
                return Array(floatBuffer)
            }
            DispatchQueue.main.async {
                let prediction = results[0]
                let label = (prediction > 0.4  && prediction < 0.6) ? "Dog 🐶" : prediction > 0.6 ? "Other Object 🤖" : "Cat 🐱"
                self.lblResult.isHidden = false
                //                    self.lblResult.text = "Predicted: \(label) (Confidence: \(prediction))"
                self.lblResult.text = "Predicted: \(label)"
            }
        } catch {
            print("Inference error: \(error)")
        }
    }
    
    @IBAction func UploadImage(_ sender: UIButton) {
        showPicker()
    }
    
}

// Convert UIImage to Pixel Buffer: To convert the UIImage into a CVPixelBuffer, which TensorFlow Lite uses for image input. Here’s a utility function to convert the image:
extension UIImage {
    func resize(to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, 1.0)
        draw(in: CGRect(origin: .zero, size: size))
        let resized = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return resized
    }
    
    func normalizedRGBData() -> Data? {
        guard let cgImage = self.cgImage else { return nil }
        
        let width = cgImage.width
        let height = cgImage.height
        let bytesPerPixel = 4
        let bytesPerRow = bytesPerPixel * width
        let bitsPerComponent = 8
        
        guard let colorSpace = cgImage.colorSpace else { return nil }
        
        guard let context = CGContext(data: nil,
                                      width: width,
                                      height: height,
                                      bitsPerComponent: bitsPerComponent,
                                      bytesPerRow: bytesPerRow,
                                      space: colorSpace,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else {
            return nil
        }
        
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        guard let imageData = context.data else { return nil }
        
        let pixelBuffer = imageData.bindMemory(to: UInt8.self, capacity: width * height * bytesPerPixel)
        
        var floatArray: [Float] = []
        
        for y in 0..<height {
            for x in 0..<width {
                let offset = (y * bytesPerRow) + (x * bytesPerPixel)
                let r = Float(pixelBuffer[offset]) / 255.0
                let g = Float(pixelBuffer[offset + 1]) / 255.0
                let b = Float(pixelBuffer[offset + 2]) / 255.0
                floatArray.append(contentsOf: [r, g, b])
            }
        }
        return Data(buffer: UnsafeBufferPointer(start: floatArray, count: floatArray.count))
    }
}
