import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *


class ConditionalVAE(BaseVAE):


    def __init__(self,
                 latent_dim: int,
                 in_channels: int = None,
                 out_channels: int = None,
                 hidden_dims: List = None,
                 img_size: int = None,
                 **kwargs) -> None:
        super(ConditionalVAE, self).__init__()

        if in_channels is None:
            in_channels = 16 # Max macros 15 -> chip layout dims, 1 heat map

        if img_size is None:
            img_size = 64

        self.latent_dim = latent_dim

        initial_in_channels = in_channels

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        dummy_input = torch.zeros(1, initial_in_channels, img_size, img_size)
        with torch.no_grad():
            dummy_output = self.encoder(dummy_input)

        self.final_conv_shape = dummy_output.shape[1:]
        flattened_hidden_size = dummy_output.view(1, -1).size(1)

        self.fc_mu = nn.Linear(flattened_hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(flattened_hidden_size, latent_dim)


        # Build Decoder
        modules = []
        flat_img_size = img_size * img_size
        self.decoder_input = nn.Linear(latent_dim + flat_img_size, flattened_hidden_size)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels=out_channels, # Need channel for each possible 
                                      kernel_size= 3, padding= 1),
                            nn.Sigmoid())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_logvar(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, *self.final_conv_shape)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Log of the variance of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, condition: Tensor) -> List[Tensor]:
        encoder_input = torch.cat([input, condition], dim=1)
        mu, log_var = self.encode(encoder_input)
        z = self.reparameterize(mu, log_var)

        B = condition.size(0)
        flat_condition = condition.view(B, -1)
        decoder_input = torch.cat([z, flat_condition], dim=1)
        return  [self.decode(decoder_input), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        r"""
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self, num_samples: int, current_device: torch.device, condition: Tensor, **kwargs) -> Tensor:
        """
        Generates new chip layouts based on a target heat map condition.
        """
        # 1. Sample random noise for z
        z = torch.randn(num_samples, self.latent_dim).to(current_device)

        # 2. Flatten the condition safely
        B = condition.size(0)
        flat_condition = condition.view(B, -1)

        # 3. Concatenate and decode (Outputs the deterministic mean)
        decoder_input = torch.cat([z, flat_condition], dim=1)
        samples = self.decode(decoder_input)
        
        return samples

    def generate(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        """
        Given an input layout and condition, returns the reconstructed layout.
        (Used primarily by TensorBoard to visualize model accuracy)
        """
        # self.forward returns: [reconstruction, input, mu, log_var]
        # We just want index 0!
        return self.forward(input, condition)[0]