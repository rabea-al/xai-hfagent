

<p align="center">
  <a href="https://github.com/XpressAI/xircuits/tree/master/xai_components#xircuits-component-library-list">Component Libraries</a> •
  <a href="https://github.com/XpressAI/xircuits/tree/master/project-templates#xircuits-project-templates-list">Project Templates</a>
  <br>
  <a href="https://xircuits.io/">Docs</a> •
  <a href="https://xircuits.io/docs/Installation">Install</a> •
  <a href="https://xircuits.io/docs/category/tutorials">Tutorials</a> •
  <a href="https://xircuits.io/docs/category/developer-guide">Developer Guides</a> •
  <a href="https://github.com/XpressAI/xircuits/blob/master/CONTRIBUTING.md">Contribute</a> •
  <a href="https://www.xpress.ai/blog/">Blog</a> •
  <a href="https://discord.com/invite/vgEg2ZtxCw">Discord</a>
</p>

<p align="center"><i>Xircuits Component Library for Hugging Face – Build advanced workflows with powerful AI tools.</i></p>

---

## Xircuits Component Library for Hugging Face Agent Toolkit

This library integrates Hugging Face capabilities into Xircuits workflows, enabling the use of language models, custom tool creation, and text/image processing.

## Table of Contents

- [Preview](#preview)
- [Prerequisites](#prerequisites)
- [Main Xircuits Components](#main-xircuits-components)
- [Try the Examples](#try-the-examples)
- [Installation](#installation)

## Preview

### SimpleHfAgent Example

<img src="https://github.com/user-attachments/assets/20bd1e8a-4bca-49f6-84bd-8e9b8843ae55" alt="HF_sample_example"/>

### Result

<img src="https://github.com/user-attachments/assets/e8fc0644-e164-4b9b-82e6-73f452cb4d02" alt="HF_sample"/>

## Prerequisites

Before you begin, you will need:

1. Python 3.9+.
2. Xircuits installed.
3. Hugging Face API token 


## Main Xircuits Components

### HfAgentInit Component:
Initializes a Hugging Face language model agent with support for tools and token configuration.

<img src="https://github.com/user-attachments/assets/58f43acb-3c89-4d53-b5dd-35d3715230dd" alt="HfAgentInit" width="150" height="125" />

### HfAgentMakeTool Component:
Creates custom tools for Hugging Face agents with a name, description, and functionality.

<img src="https://github.com/user-attachments/assets/045023e0-e723-46b6-b7fb-01e9e84b6a64" alt="HfAgentMakeTool" width="150" height="125" />

### HfAgentRun Component:
Executes prompts using the initialized agent and retrieves generated responses.

### HfReadImage Component:
Loads an image file and prepares it for AI processing.

## Try The Examples

We have provided an example workflow to help you get started with the Hugging Face Agent component library. Give it a try and see how you can create custom Hugging Face Agent components for your applications.

### HF_sample Example 

Check out the `HF_sample.xircuits` workflow. This example initializes a Hugging Face agent and uses the `HfAgentRun` component to process a prompt and generate a response. A custom tool is created and integrated with the agent to answer simple questions like "What is the capital of France?"

## Installation
To use this component library, ensure that you have an existing [Xircuits setup](https://xircuits.io/docs/main/Installation). You can then install the Hugging Face Agent library using the [component library interface](https://xircuits.io/docs/component-library/installation#installation-using-the-xircuits-library-interface), or through the CLI using:

```
xircuits install hfagent
```
You can also do it manually by cloning and installing it:
```
# base Xircuits directory
git clone https://github.com/XpressAI/xai-hfagent xai_components/xai_hfagent
pip install -r xai_components/xai_hfagent/requirements.txt 
```