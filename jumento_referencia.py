from typing import List, Optional, Tuple, TypedDict
import fire
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

# Imports antigos
# from llama.model import ModelArgs, Transformer
# from llama.tokenizer import ChatFormat, Dialog, Message, Tokenizer

# Novos imports
from llama_models.llama3_1.api import ModelArgs
from llama_models.llama3_1.api import Transformer
from llama_models.llama3_1.api import Tokenizer

class CompletionPrediction(TypedDict, total=False):
    """
    Representa uma previsão de conclusão gerada pelo modelo.

    Attributes:
        generation (str): Texto gerado pela conclusão.
        tokens (List[str]): Tokens gerados (não obrigatório).
        logprobs (List[float]): Probabilidades logarítmicas dos tokens (não obrigatório).
    """
    generation: str
    tokens: List[str]  # não obrigatório
    logprobs: List[float]  # não obrigatório

class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
    ) -> "Llama":
        """
        Cria uma instância do Llama inicializando e carregando um ponto de verificação do modelo.

        Args:
            ckpt_dir (str): Caminho para o diretório contendo arquivos de ponto de verificação.
            tokenizer_path (str): Caminho para o arquivo do tokenizador.
            max_seq_len (int): Comprimento máximo da sequência para o texto de entrada.
            max_batch_size (int): Tamanho máximo do lote para inferência.
            model_parallel_size (Optional[int], opcional): Número de processos paralelos do modelo.
                Se não fornecido, é determinado a partir do ambiente. O padrão é None.
            seed (int): Semente para inicialização do gerador de números aleatórios. O padrão é 1.

        Returns:
            Llama: Uma instância da classe Llama com o modelo e tokenizador carregados.

        Raises:
            AssertionError: Se não houver arquivos de ponto de verificação no diretório especificado,
                ou se o tamanho do paralelo do modelo não corresponder ao número de arquivos de ponto de verificação.

        Note:
            Este método inicializa o grupo de processos distribuídos, define o dispositivo como CUDA,
            e carrega o modelo pré-treinado e o tokenizador.
        """
        assert 1 <= max_seq_len <= 8192, f"max_seq_len deve estar entre 1 e 8192, obtido {max_seq_len}."
        assert os.path.isdir(ckpt_dir), f"O diretório de pontos de verificação '{ckpt_dir}' não existe."
        assert os.path.isfile(tokenizer_path), f"Arquivo do tokenizador '{tokenizer_path}' não existe."

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # A semente deve ser a mesma em todos os processos
        torch.manual_seed(seed)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"Nenhum arquivo de ponto de verificação encontrado em {ckpt_dir}"
        assert model_parallel_size == len(
            checkpoints
        ), f"Carregando um ponto de verificação para MP={len(checkpoints)} mas o tamanho do mundo é {model_parallel_size}"
        ckpt_path = checkpoints[get_model_parallel_rank()]
        # AK: adicionado weights_only=True para evitar aviso
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        assert model_args.vocab_size == tokenizer.n_words
        if torch.cuda.is_bf16_supported():
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        else:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Carregado em {time.time() - start_time:.2f} segundos")

        return Llama(model, tokenizer)

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        """
        Inicializa a instância do Llama com o modelo e tokenizador fornecidos.

        Args:
            model (Transformer): O modelo Transformer carregado.
            tokenizer (Tokenizer): O tokenizador carregado.
        """
        self.model = model
        self.tokenizer = tokenizer
        # AK: Removido toda a parte de chat por enquanto
        # self.formatter = ChatFormat(tokenizer)

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        sample_rng: torch.Generator,
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Gera sequências de texto com base nos prompts fornecidos usando o modelo de geração de linguagem.

        Args:
            prompt_tokens (List[List[int]]): Lista de prompts tokenizados, onde cada prompt é representado como uma lista de inteiros.
            max_gen_len (int): Comprimento máximo da sequência de texto gerada.
            temperature (float, opcional): Valor de temperatura para controlar a aleatoriedade na amostragem. O padrão é 0.6.
            top_p (float, opcional): Limite de probabilidade top-p para amostragem por núcleo. O padrão é 0.9.
            logprobs (bool, opcional): Flag indicando se as probabilidades logarítmicas dos tokens devem ser calculadas. O padrão é False.
            echo (bool, opcional): Flag indicando se os tokens do prompt devem ser incluídos na saída gerada. O padrão é False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: Uma tupla contendo sequências de tokens gerados e, se logprobs for True, as respectivas probabilidades logarítmicas dos tokens.

        Note:
            Este método usa os prompts fornecidos como base para gerar texto. Utiliza amostragem por núcleo para produzir texto com aleatoriedade controlada.
            Se logprobs for True, as probabilidades logarítmicas dos tokens são calculadas para cada token gerado.
        """
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens

 != pad_id

        if min_prompt_len == total_len:
            logits = self.model.forward(tokens, prev_pos)
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )

        stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens))

        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p, generator=sample_rng)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # substituir token somente se o prompt já foi gerado
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                torch.isin(next_token, stop_tokens)
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break

        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cortar para o comprimento máximo de geração
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cortar após o token eos, se houver
            for stop_token in self.tokenizer.stop_tokens:
                try:
                    eos_idx = toks.index(stop_token)
                    toks = toks[:eos_idx]
                    probs = probs[:eos_idx] if logprobs else None
                except ValueError:
                    pass
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if logprobs else None)

    def text_completion(
        self,
        prompts: List[str],
        sample_rng: torch.Generator,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> List[CompletionPrediction]:
        """
        Realiza a conclusão de texto para uma lista de prompts usando o modelo de geração de linguagem.

        Args:
            prompts (List[str]): Lista de prompts de texto para conclusão.
            temperature (float, opcional): Valor de temperatura para controlar a aleatoriedade na amostragem. O padrão é 0.6.
            top_p (float, opcional): Limite de probabilidade top-p para amostragem por núcleo. O padrão é 0.9.
            max_gen_len (Optional[int], opcional): Comprimento máximo da sequência de conclusão gerada.
                Se não fornecido, é definido como o comprimento máximo da sequência do modelo menos 1.
            logprobs (bool, opcional): Flag indicando se as probabilidades logarítmicas dos tokens devem ser calculadas. O padrão é False.
            echo (bool, opcional): Flag indicando se os tokens do prompt devem ser incluídos na saída gerada. O padrão é False.

        Returns:
            List[CompletionPrediction]: Lista de previsões de conclusão, cada uma contendo a conclusão de texto gerada.

        Note:
            Este método gera conclusões de texto para os prompts fornecidos, empregando amostragem por núcleo para introduzir aleatoriedade controlada.
            Se logprobs for True, as probabilidades logarítmicas dos tokens são calculadas para cada token gerado.
        """
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            sample_rng=sample_rng,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )
        if logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(t),
                    "tokens": [self.tokenizer.decode([x]) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]

def sample_top_p(probs, p, generator):
    """
    Realiza amostragem top-p (núcleo) em uma distribuição de probabilidade.

    Args:
        probs (torch.Tensor): Tensor de distribuição de probabilidade.
        p (float): Limite de probabilidade para amostragem top-p.

    Returns:
        torch.Tensor: Índices dos tokens amostrados.

    Note:
        A amostragem top-p seleciona o menor conjunto de tokens cuja massa de probabilidade cumulativa
        excede o limiar p. A distribuição é renormalizada com base nos tokens selecionados.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1, generator=generator)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
):
    """
    Exemplos de execução com os modelos pré-treinados (sem ajuste fino). Os prompts são
    geralmente na forma de um prefixo de texto incompleto que o modelo pode tentar completar.

    A janela de contexto dos modelos llama3 é de 8192 tokens, portanto `max_seq_len` deve ser <= 8192.
    `max_gen_len` é necessário porque os modelos pré-treinados geralmente não param conclusões naturalmente.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # AK: corrigido um bug de espaço em branco final e ajustados os prompts
    prompts: List[str] = [
        # Para esses prompts, a resposta esperada é a continuação natural do prompt
        "Claramente, o significado da vida é",
        "Simplificando, a teoria da relatividade afirma que",
        """O repositório llm.c no GitHub é""",
        # Prompt com poucos exemplos (fornecendo alguns exemplos antes de pedir ao modelo para completar mais);
        """Traduza inglês para francês:

        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
    ]
    sample_rng = torch.Generator(device='cuda')
    sample_rng.manual_seed(1337)
    results = generator.text_completion(
        prompts,
        sample_rng=sample_rng,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    for prompt, result in zip(prompts, results):
        print(prompt, end="") # AK: alterado end="\n" para end=""
        print(f"{result['generation']}")
        print("\n==================================\n")

    # AK: adicionado limpeza do torch.distributed
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    fire.Fire(main)
