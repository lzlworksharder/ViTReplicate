import torch
from torchmetrics import Accuracy, MeanMetric
from accelerate import Accelerator
import os
from datetime import datetime
from tqdm import tqdm

class TrainLoop:
    def __init__(self,
            train_loader, val_loader,
            model, scheduler, optimizer, 
            steps,
            num_classes=100,
            eval_step=None, 
            # gradient_accumulation_steps=1, 
            # precision='bf16', 
            log_step=100, 
            save_step=None, 
            debug=False,
            trial=None,
            early_stopping=True,
            patience=5,  
            min_delta=0.001,
            ):  
        # accelerator with multi-GPU support
        self.accelerator = Accelerator(
            # gradient_accumulation_steps=gradient_accumulation_steps,
            # mixed_precision=precision,
            log_with='tensorboard',
            project_dir='./logs',
        )
        
        # Only initialize trackers on main process
        if self.accelerator.is_local_main_process:
            self.accelerator.init_trackers(f"res50_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        
        # 初始化指标
        self.acc = Accuracy(task="multiclass", num_classes=num_classes).to(self.accelerator.device)
        self.loss_metric = MeanMetric().to(self.accelerator.device)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes).to(self.accelerator.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.train_loader, self.val_loader, self.model, self.optimizer, self.scheduler  = self.accelerator.prepare(
            self.train_loader, self.val_loader, self.model, self.optimizer, self.scheduler
        )
        self.cur_step = 0
        
        self.steps = steps
        self.debug = debug
        self.eval_step = eval_step if eval_step is not None else steps
        self.save_step = save_step
        self.log_step = log_step
        self.trial = trial

        # 早停相关参数
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_acc = 0
        self.patience_counter = 0
        self.save_dir = "./ckpt"
        self.early_stopping = early_stopping

    def run_loop(self):
        # Only show progress bar on main process
        tbar = tqdm(total=self.steps, desc="Training", position=0, disable=not self.accelerator.is_local_main_process)
        while self.cur_step<self.steps:
            for batch in self.train_loader:
                # a step
                self.acc.reset()
                self.loss_metric.reset()
                self.model.train()

                # 用ds+梯度累计时不能写上下文管理器，accumulate的no_sync和zero分片的all reduce冲突！
                # with self.accelerator.accumulate(self.model):

                # forward + backward
                output = self.model(**batch) if isinstance(batch, dict) else self.model(*batch)
                loss = output.loss
                logits = output.logits
                preds = torch.argmax(logits, dim=-1)
                self.accelerator.backward(loss)

                # gather results and update metrics
                all_loss = self.accelerator.gather_for_metrics(loss)
                all_preds, all_labels = self.accelerator.gather_for_metrics((preds, batch['labels'] if isinstance(batch, dict) else batch[1]))
                self.acc.update(all_preds, all_labels)
                self.loss_metric.update(all_loss)


                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.cur_step += 1
                    tbar.update()
                    self.optimizer.step()
                    if self.scheduler.scheduler.__class__.__name__ == "ReduceLROnPlateau":
                        self.scheduler.step(self.acc.compute())
                    else:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
            
                    if self.cur_step % self.log_step == 0:
                        # Gather metrics from all processes
                        train_loss = self.loss_metric.compute().item()
                        train_acc = self.acc.compute().item()

                        metrics = {
                            "train_loss": train_loss,
                            "train_acc": train_acc,
                        }
                        
                        if self.accelerator.is_local_main_process:
                            self.accelerator.log(metrics, step=self.cur_step)
                            tbar.set_postfix({'ls': metrics['train_loss'], 'acc': metrics['train_acc']})

                    if self.cur_step % self.eval_step == 0:
                        self.model.eval()
                        with torch.inference_mode():
                            metrics = self.val()
                            val_acc = metrics['val_acc']
                            
                            if self.accelerator.is_local_main_process:
                                self.accelerator.log(metrics, step=self.cur_step)
                            # Synchronize validation results across all processes
                            self.accelerator.wait_for_everyone()
                            if val_acc > (self.best_val_acc + self.min_delta):
                                self.best_val_acc = val_acc
                                self.patience_counter = 0
                                if self.accelerator.is_local_main_process:
                                    save_path = os.path.join(self.save_dir, f"best_model_step_{self.cur_step}")
                                    os.makedirs(self.save_dir, exist_ok=True)
                                    # self.accelerator.save_state(save_path)
                                    self.accelerator.print(f"\nStep {self.cur_step}: New best val_acc: {val_acc:.4f}")
                            else:
                                if self.accelerator.is_local_main_process:
                                    self.accelerator.print(f"\nStep {self.cur_step}: val_acc: {val_acc:.4f}")
                                self.patience_counter += 1                           
                                if self.early_stopping and self.patience_counter >= self.patience:
                                    if self.accelerator.is_local_main_process:
                                        self.accelerator.print(f"Early stopping at step {self.cur_step}.")
                                    return

                        if self.trial is not None:
                            self.trial.report(metrics['val_acc'], step=self.cur_step)
                            if self.trial.should_prune():
                                from optuna import TrialPruned
                                raise TrialPruned()
                    
                    if self.save_step is not None and self.cur_step % self.save_step == 0:
                        self.accelerator.wait_for_everyone()
                        if self.accelerator.is_local_main_process:
                            save_path = os.path.join("./ckpt/step_{}".format(self.cur_step))
                            os.makedirs(os.path.dirname(save_path), exist_ok=True)
                            self.accelerator.save_state(save_path)

                    if self.cur_step >= self.steps:
                        break
        
        if self.accelerator.is_local_main_process:
            self.accelerator.end_training()
        
    def val(self):
        self.val_acc.reset()

        for i,batch in enumerate(self.val_loader):
            outputs = self.model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds,all_labels = self.accelerator.gather_for_metrics((preds, batch['labels'])) 
            self.val_acc.update(all_preds, all_labels)

            if self.debug and i>= 2: break

        metrics = {
            "val_acc": self.val_acc.compute().item(),
        }
        return metrics

