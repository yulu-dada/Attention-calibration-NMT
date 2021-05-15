import os, torch
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from fairseq.data import data_utils, PrependTokenDataset, LanguagePairDataset, ConcatDataset

import logging
logger = logging.getLogger(__name__)

@register_task('attack_translation_task')
class AttackTranslationTask(TranslationTask):

    def add_args(parser):
        TranslationTask.add_args(parser)
        pass

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)
        # model.set_stage(self.args.training_stage)
        # criterion.set_stage(self.args.training_stage)
        
        with torch.autograd.profiler.record_function("forward"):
            loss, mask_loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
            mask_loss *=0

        with torch.autograd.profiler.record_function("backward"):
            for name, p in model.named_parameters():
                if 'attacker' in name: 
                    p.requires_grad = False
                else:
                    p.requires_grad = True
            optimizer.backward(loss, retain_graph=True)
            for name, p in model.named_parameters():
                if 'attacker' in name: 
                    p.requires_grad = True
                else:
                    p.requires_grad = False
            optimizer.backward(mask_loss)
            for name, p in model.named_parameters():
                p.requires_grad = True
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, mask_loss, sample_size, logging_output = criterion(model, sample)
        # loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.args.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            return generator.generate(
                models, sample, prefix_tokens=prefix_tokens, constraints=constraints
            )

