from transformerocr.tool.translate import translate

def visualize(model, vocab, gen, sample=30):
    pred_sents = []
    actual_sents = []
    img_files = []
    n = 0

    for batch in  valid_loader.gen(sample):
	translated_sentence = translate(batch['img'], model)
	pred_sent = vocab.batch_decode(translated_sentence.tolist())
	actual_sent = vocab.batch_decode(batch['tgt_input'].T.tolist())
	
	img_files.extend(batch['filenames'])
	
	pred_sents.extend(pred_sent)
	actual_sents.extend(actual_sent)
        
        n += len(actual_sents)
        if n > sample: break


    for vis_idx in range(0, len(img_files)): 
	img_path = img_files[vis_idx]
	pred_sent = pred_sents[vis_idx]
	actual_sent = actual_sents[vis_idx]

	img = Image.open(open(img_path, 'rb'))
	plt.imshow(img)
	plt.title('pred: {} - actual: {}'.format(pred_sent, actual_sent))
	plt.show()
