
def translate(img, model):
    "data: BxCXHxW"
    model.eval()
    device = img.device
    
    with torch.no_grad():
        translated_sentence = [[1]*len(img)]

        while not all(np.any(np.asarray(translated_sentence).T==2, axis=1)):

            tgt_inp = torch.LongTensor(translated_sentence).to(device)
            
            output = model(img, tgt_inp, tgt_key_padding_mask=None)
            output = output.to('cpu')

            values, indices  = torch.topk(output, 5)

            indices = indices[:, -1, 0]
            indices = indices.tolist()

            translated_sentence.append(indices)    
        translated_sentence = np.asarray(translated_sentence).T
    
    model.train()
    
    return translated_sentence
