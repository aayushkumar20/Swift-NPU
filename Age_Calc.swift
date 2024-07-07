import UIKit
import AVFoundation
import Vision
import CoreML

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {

    var captureSession: AVCaptureSession!
    var previewLayer: AVCaptureVideoPreviewLayer!
    var ageLabel: UILabel!

    override func viewDidLoad() {
        super.viewDidLoad()

        // Set up the camera preview
        captureSession = AVCaptureSession()
        guard let videoCaptureDevice = AVCaptureDevice.default(for: .video) else { return }
        let videoInput: AVCaptureDeviceInput

        do {
            videoInput = try AVCaptureDeviceInput(device: videoCaptureDevice)
        } catch {
            return
        }

        if (captureSession.canAddInput(videoInput)) {
            captureSession.addInput(videoInput)
        } else {
            return
        }

        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        if (captureSession.canAddOutput(videoOutput)) {
            captureSession.addOutput(videoOutput)
        } else {
            return
        }

        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.frame = view.layer.bounds
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)

        captureSession.startRunning()

        // Set up the age label
        ageLabel = UILabel()
        ageLabel.text = "Age: Unknown"
        ageLabel.textColor = .white
        ageLabel.font = UIFont.systemFont(ofSize: 24)
        ageLabel.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(ageLabel)

        NSLayoutConstraint.activate([
            ageLabel.bottomAnchor.constraint(equalTo: view.bottomAnchor, constant: -20),
            ageLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor)
        ])
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        // Create a request handler
        let requestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])

        // Create a request
        let request = VNDetectFaceRectanglesRequest { [weak self] (request, error) in
            if let results = request.results as? [VNFaceObservation], let firstResult = results.first {
                self?.handleFaceObservation(firstResult, in: pixelBuffer)
            }
        }

        // Perform the request
        try? requestHandler.perform([request])
    }

    func handleFaceObservation(_ observation: VNFaceObservation, in pixelBuffer: CVPixelBuffer) {
        // Convert VNFaceObservation to CGRect
        let boundingBox = observation.boundingBox
        let faceRect = VNImageRectForNormalizedRect(boundingBox, Int(CVPixelBufferGetWidth(pixelBuffer)), Int(CVPixelBufferGetHeight(pixelBuffer)))

        // Crop the face image from the pixel buffer
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer).cropped(to: faceRect)

        // Convert CIImage to UIImage
        let context = CIContext()
        if let cgImage = context.createCGImage(ciImage, from: ciImage.extent) {
            let uiImage = UIImage(cgImage: cgImage)

            // Perform age estimation
            estimateAge(from: uiImage)
        }
    }

    func estimateAge(from image: UIImage) {
        // Load the age prediction model
        guard let model = try? VNCoreMLModel(for: AgePredictionModel().model) else { return }

        // Create a request
        let request = VNCoreMLRequest(model: model) { [weak self] (request, error) in
            if let results = request.results as? [VNClassificationObservation], let firstResult = results.first {
                DispatchQueue.main.async {
                    self?.ageLabel.text = "Age: \(firstResult.identifier)"
                }
            }
        }

        // Convert UIImage to CVPixelBuffer
        guard let pixelBuffer = image.toCVPixelBuffer() else { return }

        // Create a request handler
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])

        // Perform the request
        try? handler.perform([request])
    }
}

extension UIImage {
    func toCVPixelBuffer() -> CVPixelBuffer? {
        let width = Int(self.size.width)
        let height = Int(self.size.height)
        let attrs = [kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
                     kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!] as CFDictionary
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32ARGB, attrs, &pixelBuffer)
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else { return nil }

        CVPixelBufferLockBaseAddress(buffer, [])
        let pixelData = CVPixelBufferGetBaseAddress(buffer)

        let rgbColorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: pixelData, width: width, height: height, bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer), space: rgbColorSpace, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)

        context?.draw(self.cgImage!, in: CGRect(x: 0, y: 0, width: width, height: height))

        CVPixelBufferUnlockBaseAddress(buffer, [])

        return buffer
    }
}

// Code for ViewController.Swift