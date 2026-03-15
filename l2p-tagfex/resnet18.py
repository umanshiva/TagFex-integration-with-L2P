import torch
import torch.nn as nn
from torchvision.models import resnet18
from prompt import Prompt
# Reuse the same Prompt class from your ViT implementation
# from your_code.prompt import Prompt


class ResNet18Prompted(nn.Module):
    def __init__(
        self,
        num_classes=100,
        prompt_length=7,
        prompt_pool=True,
        pool_size=15,
        top_k=5,
        embedding_key='cls',
        prompt_key=True,
        shared_prompt_key=False,
        head_type='token+prompt',  # same as ViT options
        use_prompt_mask=False,
        proj_dim=512,
        prompt_init='uniform',
    ):
        super().__init__()

        # === Backbone (pretrained ResNet-18) ===
        self.backbone = resnet18(pretrained=True)
        self.embed_dim = self.backbone.fc.in_features  # Usually 512
        self.backbone.fc = nn.Identity()  # remove final classification head

        self.num_classes = num_classes
        self.head_type = head_type
        self.prompt_pool = prompt_pool
        self.prompt_length = prompt_length
        self.top_k = top_k
        self.use_prompt_mask = use_prompt_mask

        # === Prompt module ===
        if prompt_pool:
            self.prompt = Prompt(
                length=prompt_length, embed_dim=self.embed_dim, pool_size=pool_size,
                top_k=top_k, embedding_key=embedding_key, prompt_key=prompt_key,
                batchwise_prompt=True,
                prompt_init=prompt_init
            )

        # # === ResNet18 doesn't use class tokens, but we mirror ViT interface ===
        # if cls_token:
        #     self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        #     nn.init.trunc_normal_(self.cls_token, std=0.02)

        # === Normalization + Head ===
        self.fc_norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, task_id=-1, cls_features=None, train=False):
        B = x.shape[0]

        # Extract backbone features
        feats = self.backbone(x)                # (B, 512)
        feats = feats.unsqueeze(1)              # (B, 1, 512)
        tokens = feats  # The main token

        res = {}

        # === Handle Prompt Module (same API as in ViT) ===
        if self.prompt_pool:
            prompt_out = self.prompt(
                x_embed=tokens,  # use backbone features as input to prompt selector
                cls_features=cls_features
            )
            prompt_emb = prompt_out['prompted_embedding']  # (B, K*P, 512)
            total_prompt_len = self.prompt_length * self.top_k

            # Append prompts to main token
            tokens = torch.cat((tokens, prompt_emb), dim=1)  # (B, 1+K*P, 512)

            # Save prompt info in result
            res.update({
                'prompt_idx': prompt_out.get('prompt_idx'),
                'prompt_loss': prompt_out.get('prompt_loss'),
                'prompt_mask': prompt_out.get('prompt_mask'),
                'total_prompt_len': total_prompt_len
            })
        else:
            res.update({
                'prompt_idx': None,
                'prompt_loss': None,
                'prompt_mask': None,
                'total_prompt_len': 0
            })

        # Add cls token if used
        # if self.class_token:
        #     tokens = torch.cat((self.cls_token.expand(B, -1, -1), tokens), dim=1)  # (B, 1+N, 512)

        res['x'] = tokens  # save full token sequence for forward_head
        return res

    def forward_head(self, res, pre_logits=False):
        x = res['x']  # (B, N, 512)

        if self.head_type == 'token' and self.class_token:
            pooled = x[:, 0]  # first is cls
        elif self.head_type == 'gap':
            pooled = x.mean(dim=1)
        elif self.head_type == 'prompt' and self.prompt_pool:
            pooled = x[:, 1:(1 + res['total_prompt_len'])].mean(dim=1)
        elif self.head_type == 'token+prompt' and self.prompt_pool:
            pooled = x.mean(dim=1)  # mean of all tokens
        else:
            raise ValueError(f'Invalid head_type: {self.head_type}')

        res['pre_logits'] = pooled

        x_norm = self.fc_norm(pooled)
        logits = self.head(x_norm)

        res['logits'] = logits
        return res

    def forward(self, x, task_id=-1, cls_features=None, train=False):
        res = self.forward_features(x, task_id=task_id, cls_features=cls_features, train=train)
        res = self.forward_head(res)
        return res


if __name__ == "__main__":
    model = ResNet18Prompted(
        num_classes=100, prompt_length=3
    ).to('cuda')

    dummy_input = torch.randn(2, 3, 32, 32)  # batch of 2 images
    from torchinfo import summary
    summary(model, (2, 3, 32, 32))
