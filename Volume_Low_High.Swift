import Cocoa
import AVFoundation
import Vision

class ViewController: NSViewController, AVCaptureVideoDataOutputSampleBufferDelegate {

    var captureSession: AVCaptureSession!
    var previewLayer: AVCaptureVideoPreviewLayer!
    var lastPinchDistance: CGFloat = 0

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
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        let requestHandler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        let request = VNDetectHumanHandPoseRequest { [weak self] request, error in
            if let results = request.results as? [VNHumanHandPoseObservation], let observation = results.first {
                self?.handleHandPoseObservation(observation)
            }
        }

        do {
            try requestHandler.perform([request])
        } catch {
            print(error)
        }
    }

    func handleHandPoseObservation(_ observation: VNHumanHandPoseObservation) {
        guard let thumbTip = try? observation.recognizedPoints(.all)[.thumbTip],
              let indexTip = try? observation.recognizedPoints(.all)[.indexTip] else { return }

        if thumbTip.confidence > 0.3 && indexTip.confidence > 0.3 {
            let thumbTipLocation = CGPoint(x: thumbTip.location.x * view.bounds.width, y: (1 - thumbTip.location.y) * view.bounds.height)
            let indexTipLocation = CGPoint(x: indexTip.location.x * view.bounds.width, y: (1 - indexTip.location.y) * view.bounds.height)

            let pinchDistance = hypot(thumbTipLocation.x - indexTipLocation.x, thumbTipLocation.y - indexTipLocation.y)

            if lastPinchDistance == 0 {
                lastPinchDistance = pinchDistance
                return
            }

            let volumeChange = pinchDistance - lastPinchDistance
            adjustSystemVolume(by: volumeChange / 100.0) // Adjust sensitivity here

            lastPinchDistance = pinchDistance
        }
    }

    func adjustSystemVolume(by amount: CGFloat) {
        let volumeStep: Float = 0.05 // Change this value to adjust the sensitivity
        let volumeChange = Float(amount) * volumeStep

        var defaultOutputDeviceID = AudioObjectID()
        var propertySize = UInt32(MemoryLayout.size(ofValue: defaultOutputDeviceID))

        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultOutputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMaster
        )

        let result = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &address,
            0,
            nil,
            &propertySize,
            &defaultOutputDeviceID
        )

        guard result == noErr else { return }

        address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwareServiceDeviceProperty_VirtualMasterVolume,
            mScope: kAudioDevicePropertyScopeOutput,
            mElement: kAudioObjectPropertyElementMaster
        )

        var currentVolume = Float(0)
        propertySize = UInt32(MemoryLayout.size(ofValue: currentVolume))

        AudioObjectGetPropertyData(defaultOutputDeviceID, &address, 0, nil, &propertySize, &currentVolume)

        var newVolume = currentVolume + volumeChange
        newVolume = min(max(newVolume, 0), 1) // Clamp between 0 and 1

        AudioObjectSetPropertyData(defaultOutputDeviceID, &address, 0, nil, propertySize, &newVolume)
    }
}

// ViewController.Swift