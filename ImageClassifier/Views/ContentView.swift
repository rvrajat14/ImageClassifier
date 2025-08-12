//
//  ContentView.swift
//  ImageClassifier
//
//  Created by Rajat Verma on 12/08/25.
//

import SwiftUI
import CoreML
import Vision

struct ContentView: View {
    @State private var image: UIImage?
    @State private var classificationLabel = "Select or capture an image"
    @State private var showingImagePicker = false
    @State private var inputImage: UIImage?
    @State private var isClassifying = false
    @State private var animateLabel = false

    var body: some View {
        NavigationView {
            ZStack {
                LinearGradient(
                    gradient: Gradient(colors: [Color.purple.opacity(0.8), Color.blue.opacity(0.8)]),
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing)
                .ignoresSafeArea()

                VStack(spacing: 24) {
                    Spacer()

                    if let image = image {
                        Image(uiImage: image)
                            .resizable()
                            .scaledToFit()
                            .frame(maxHeight: 300)
                            .cornerRadius(20)
                            .shadow(radius: 12)
                            .transition(.opacity)
                    } else {
                        RoundedRectangle(cornerRadius: 20)
                            .fill(Color.white.opacity(0.2))
                            .frame(height: 300)
                            .overlay(
                                Text("No Image Selected")
                                    .foregroundColor(.white.opacity(0.7))
                                    .font(.title3)
                            )
                    }

                    if isClassifying {
                        ProgressView("Classifying...")
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                            .foregroundColor(.white)
                    }

                    Text(classificationLabel)
                        .font(.title2.weight(.semibold))
                        .foregroundColor(.white)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                        .opacity(animateLabel ? 1 : 0)
                        .animation(.easeInOut(duration: 0.6), value: animateLabel)

                    Button(action: { showingImagePicker = true }) {
                        Label("Select Photo", systemImage: "photo")
                            .font(.headline)
                            .padding()
                            .frame(maxWidth: .infinity)
                            .background(Color.orange)
                            .foregroundColor(.white)
                            .cornerRadius(15)
                            .shadow(radius: 8)
                    }
                    .padding(.horizontal)

                    Spacer()
                }
                .padding()
                .navigationTitle("MobileNetV2 Classifier")
                .sheet(isPresented: $showingImagePicker, onDismiss: classifyImage) {
                    ImagePicker(image: $inputImage)
                }
                .onChange(of: inputImage) { _ in classifyImage() }
            }
        }
        .navigationViewStyle(StackNavigationViewStyle())
    }

    func classifyImage() {
        guard let inputImage = inputImage else { return }
        withAnimation { isClassifying = true; animateLabel = false }
        image = inputImage
        classificationLabel = ""

        DispatchQueue.global(qos: .userInitiated).async {
            let result = runImageClassification(inputImage)
            DispatchQueue.main.async {
                withAnimation {
                    classificationLabel = result
                    isClassifying = false
                    animateLabel = true
                    UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                }
            }
        }
    }

    func runImageClassification(_ uiImage: UIImage) -> String {
        guard let ciImage = CIImage(image: uiImage) else {
            return "Failed to convert image"
        }
        
        // ✅ Updated to use init(configuration:) with error handling
        var config = MLModelConfiguration()
        config.computeUnits = .all // Use CPU, GPU, and Neural Engine
        
        guard let coreMLModel = try? MobileNetV2(configuration: config) else {
            return "Failed to load Core ML model"
        }
        
        guard let model = try? VNCoreMLModel(for: coreMLModel.model) else {
            return "Failed to create VNCoreMLModel"
        }
        
        let request = VNCoreMLRequest(model: model)
        let handler = VNImageRequestHandler(ciImage: ciImage)
        
        var classificationResult = "Unknown"
        
        do {
            try handler.perform([request])
            if let results = request.results as? [VNClassificationObservation],
               let topResult = results.first {
                let confidence = Int(topResult.confidence * 100)
                classificationResult = "\(topResult.identifier) (\(confidence)%)"
            }
        } catch {
            classificationResult = "Error: \(error.localizedDescription)"
        }
        
        return classificationResult
    }

}

#Preview {
    ContentView()
}
