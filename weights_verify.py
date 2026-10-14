from download_weights import model
with open("weights_verification.txt", "w") as f:
    for name, p in model.named_parameters(): 
        total = p.sum().item()
        first = p.T.flatten()[0].item()
        last = p.T.flatten()[-1].item()
        
        f.write(f"{name} {first:.6f} {last:.6f} {total:.6f}\n")



