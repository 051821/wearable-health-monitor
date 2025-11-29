########################################
# Terraform Providers
########################################

terraform {
  required_providers {
    null = {
      source  = "hashicorp/null"
      version = "~> 3.2"
    }

    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }

  required_version = ">= 1.1.0"
}

provider "null" {}
provider "docker" {}

########################################
# 1. DEPLOY PUPPET MODULE (mhealth)
########################################

resource "null_resource" "deploy_mhealth_puppet" {
  triggers = {
    always = timestamp()
  }

  # Upload init.pp ONLY (module file)
  provisioner "file" {
    source      = "puppet-files/init.pp"
    destination = "/tmp/init.pp"

    connection {
      type        = "ssh"
      user        = "anushka"
      private_key = file("C:/Users/91739/.ssh/id_rsa")
      host        = "192.168.241.128"
    }
  }

  provisioner "remote-exec" {
    inline = [
      # create module path
      "sudo mkdir -p /etc/puppetlabs/code/environments/production/modules/mhealth/manifests",

      # move init.pp into the module
      "sudo mv /tmp/init.pp /etc/puppetlabs/code/environments/production/modules/mhealth/manifests/init.pp",

      # validate puppet module
      "sudo /opt/puppetlabs/bin/puppet parser validate /etc/puppetlabs/code/environments/production/modules/mhealth/manifests/init.pp",

      # restart puppet server
      "sudo systemctl restart puppetserver",
      "sleep 3"
    ]

    connection {
      type        = "ssh"
      user        = "anushka"
      private_key = file("C:/Users/91739/.ssh/id_rsa")
      host        = "192.168.241.128"
    }
  }

  # Cleanup
  provisioner "remote-exec" {
    when = destroy

    inline = [
      "sudo rm -rf /etc/puppetlabs/code/environments/production/modules/mhealth",
      "sudo systemctl restart puppetserver",
      "echo 'mhealth module removed.'"
    ]

    connection {
      type        = "ssh"
      user        = "anushka"
      private_key = file("C:/Users/91739/.ssh/id_rsa")
      host        = "192.168.241.128"
    }
  }
}

########################################
# 2. DEPLOY NAGIOS CONFIG
########################################

resource "null_resource" "deploy_mhealth_nagios" {
  triggers = {
    always = timestamp()
  }

  provisioner "file" {
    source      = "nagios/mh_win.cfg"
    destination = "/tmp/mh_win.cfg"

    connection {
      type        = "ssh"
      user        = "anushka"
      private_key = file("C:/Users/91739/.ssh/id_rsa")
      host        = "192.168.241.128"
    }
  }

  provisioner "remote-exec" {
    inline = [
      "sudo mv /tmp/mh_win.cfg /usr/local/nagios/etc/objects/mh_win.cfg",

      # adds cfg_file only if not already added
      "grep -qxF 'cfg_file=/usr/local/nagios/etc/objects/mh_win.cfg' /usr/local/nagios/etc/nagios.cfg || echo 'cfg_file=/usr/local/nagios/etc/objects/mh_win.cfg' | sudo tee -a /usr/local/nagios/etc/nagios.cfg",

      # validate Nagios
      "sudo /usr/local/nagios/bin/nagios -v /usr/local/nagios/etc/nagios.cfg",

      # restart
      "sudo systemctl restart nagios"
    ]

    connection {
      type        = "ssh"
      user        = "anushka"
      private_key = file("C:/Users/91739/.ssh/id_rsa")
      host        = "192.168.241.128"
    }
  }

  provisioner "remote-exec" {
    when = destroy

    inline = [
      "sudo rm -f /usr/local/nagios/etc/objects/mh_win.cfg",
      "sudo sed -i '/mh_win.cfg/d' /usr/local/nagios/etc/nagios.cfg",
      "sudo systemctl restart nagios"
    ]

    connection {
      type        = "ssh"
      user        = "anushka"
      private_key = file("C:/Users/91739/.ssh/id_rsa")
      host        = "192.168.241.128"
    }
  }

  depends_on = [null_resource.deploy_mhealth_puppet]
}

########################################
# 3. DOCKER BUILD & RUN
########################################

resource "docker_image" "mhealth_onnx_image" {
  name = "mhealth_onnx:latest"

  build {
    context    = "../app"
    dockerfile = "../app/Dockerfile"
  }
}

resource "docker_container" "mhealth_onnx_container" {
  name  = "mhealth_onnx_container"
  image = docker_image.mhealth_onnx_image.name

  ports {
    internal = 8501
    external = 8501
  }

  restart = "always"
}
