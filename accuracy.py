from src.predict import predict

def test_model(model, filepath):
    correct = 0
    total = 0

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split('|')

            if len(parts) != 2:
                print("Skipping bad line:", line)
                continue


    accuracy = (correct / total) * 100
    print(f"Accuracy: {accuracy:.2f}%")
    
