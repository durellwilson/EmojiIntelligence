import Foundation

/// A simple feedforward neural network implementation in Swift
/// Supports customizable architecture and training parameters
public class NeuralNetwork {
    
    // MARK: - Configuration
    public struct Configuration {
        public let learningRate: Float
        public let momentum: Float
        public let maxIterations: Int
        
        public init(learningRate: Float = 0.3, momentum: Float = 0.6, maxIterations: Int = 70000) {
            self.learningRate = learningRate
            self.momentum = momentum
            self.maxIterations = maxIterations
        }
    }
    
    // MARK: - Properties
    public let configuration: Configuration
    private var layers: [Layer] = []
    
    // MARK: - Initialization
    public init(inputSize: Int, hiddenSize: Int, outputSize: Int, configuration: Configuration = Configuration()) {
        self.configuration = configuration
        self.layers = [
            Layer(inputSize: inputSize, outputSize: hiddenSize),
            Layer(inputSize: hiddenSize, outputSize: outputSize)
        ]
    }
    
    // MARK: - Forward Pass
    /// Runs a forward pass through the network
    /// - Parameter input: Input values for the network
    /// - Returns: Output activations from the final layer
    public func run(input: [Float]) -> [Float] {
        return layers.reduce(input) { activations, layer in
            layer.run(inputArray: activations)
        }
    }
    
    // MARK: - Training
    /// Trains the network using backpropagation
    /// - Parameters:
    ///   - input: Input training data
    ///   - targetOutput: Expected output values
    public func train(input: [Float], targetOutput: [Float]) {
        train(input: input, targetOutput: targetOutput, 
              learningRate: configuration.learningRate, 
              momentum: configuration.momentum)
    }
    
    /// Trains the network with custom parameters
    /// - Parameters:
    ///   - input: Input training data
    ///   - targetOutput: Expected output values
    ///   - learningRate: Learning rate for this training step
    ///   - momentum: Momentum for this training step
    public func train(input: [Float], targetOutput: [Float], learningRate: Float, momentum: Float) {
        let calculatedOutput = run(input: input)
        
        // Calculate error using functional approach
        let error = zip(targetOutput, calculatedOutput).map { target, output in
            target - output
        }
        
        // Backpropagate error through layers
        _ = layers.reversed().reduce(error) { currentError, layer in
            layer.train(error: currentError, learningRate: learningRate, momentum: momentum)
        }
    }
    
    // MARK: - Utility Methods
    /// Calculates the mean squared error for the current prediction
    /// - Parameters:
    ///   - input: Input data
    ///   - targetOutput: Expected output
    /// - Returns: Mean squared error value
    public func calculateError(input: [Float], targetOutput: [Float]) -> Float {
        let output = run(input: input)
        let errors = zip(targetOutput, output).map { target, predicted in
            pow(target - predicted, 2)
        }
        return errors.reduce(0, +) / Float(errors.count)
    }
}
