# Useful Snippets

## Anaconda/Mamba

To install packages from mamba

```bash
mamba install -c CHANNEL_NAME [PACKAGES...]
```

To export the environment changes in an OS-agnostic form:

```bash
mamba env export --from-history > environment.yml
```

## Mypy

To ignore a specific error on a line use

```python
BAD_CODE # type: ignore[<code>]
```
