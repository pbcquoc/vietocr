class Trainer():
    def __init__(self, config):
        self.model = 
       	self.epoch = 0 
        self.optimizer = ScheduledOptim(
            Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09),
            0.2, d_model, n_warmup_steps)

        self.criterion = nn.CrossEntropyLoss(ignore_index=0) 

    def train(self, train_gen, val_gen, num_epochs):
	for epoch in range(num_epochs):
	    total_loss = 0
            self.epoch = epoch
	    for idx, batch in enumerate(train_gen.gen(32)):
			
		loss = self.step(batch)
		
		total_loss += loss
		train_losses.append((idx, loss))

		if idx % print_every == print_every - 1:
		    info = 'epoch: {} iter: {} - train loss: {}'.format(epoch, idx, total_loss/print_every)
		    total_loss = 0
		
	    save_checkpoint('transformerocr_checkpoint')
	    val_loss = validate(val_gen)
	    info = 'epoch: {} - val loss: {}'.format(epoch, val_loss)

    def validate(self, valid_loader):
        self.model.eval()
    
	total_loss = []
	
	with torch.no_grad():
	    for step, batch in enumerate(valid_loader.gen(30)):

		img, tgt_input, tgt_output, tgt_padding_mask = batch['img'], batch['tgt_input'], batch['tgt_output'], batch['tgt_padding_mask']

		outputs = self.model(img, tgt_input, tgt_padding_mask)

		loss = self.criterion(rearrange(outputs, 'b t v -> (b t) v'), rearrange(tgt_output, 'b o -> (b o)'))

		total_loss.append(loss.item())
		
	total_loss = np.mean(total_loss)
	self.model.train()
	
	return total_loss

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
	start_epoch = checkpoint['epoch']
	
	optim = ScheduledOptim(
	    Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09),
	    0.2, d_model, n_warmup_steps)
	
	self.optimizer.load_state_dict(checkpoint['optimizer'])
	self.model.load_state_dict(checkpoint['state_dict'])
	

    def save_checkpoint(self, filename):
        optimizer_state_dict = {
	    'init_lr':self.optimizer.init_lr,
	    'd_model':self.optimizer.d_model,
	    'n_warmup_steps':self.optimizer.n_warmup_steps,
	    'n_steps':self.optimizer.n_steps,
	    '_optimizer':self.optimizer._optimizer.state_dict(),
	    'epoch':epoch
	}

	state = {'epoch': epoch, 'state_dict': self.model.state_dict(),
		     'optimizer': optimizer_state_dict}
	
	torch.save(state, '{}_epoch_{}.pt'.format(filename, epoch))


    def step(self, batch):
	self.model.train()
	
	img, tgt_input, tgt_output, tgt_padding_mask = batch['img'], batch['tgt_input'], batch['tgt_output'], batch['tgt_padding_mask']
	
	outputs = self.model(img, tgt_input, tgt_key_padding_mask=tgt_padding_mask)
	loss = self.criterion(rearrange(outputs, 'b t v -> (b t) v'), rearrange(tgt_output, 'b o -> (b o)'))

	self.optimizer.zero_grad()
	loss.backward()
	self.optimizer.step_and_update_lr()
	
	return loss.item()
