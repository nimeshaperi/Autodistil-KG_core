# Architecture 
The main purpose of this package is to abstract away the complexity of the following components
1. Graph Traversing
2. ChatML Dataset Creation 
3. Finetuner
4. Evaluator

The idea is to have interfaces that allow running each module independent to each other for other use cases. This will act as syntactic sugar and easy interface to work with.

The package should also allow the ability to define a set pipeline with each component configuration that can be run. This will probably be the most common usage.

Graph Traverser -> ChatML Converter -> FineTuner -> Evaluator

---

