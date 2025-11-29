class mhealth {

  # Create monitoring directory
  file { 'C:/mhealth_monitor':
    ensure => directory,
  }

  # Create health check script
  file { 'C:/mhealth_monitor/health_check.ps1':
    ensure  => file,
    content => @("EOF")
      try {
        Invoke-WebRequest -Uri http://localhost:8501 -UseBasicParsing -TimeoutSec 5
        exit 0
      } catch {
        exit 2
      }
      EOF
  }

  # Create scheduled task to auto-start Docker Desktop
  exec { 'schedule_docker_desktop_start':
    command => 'schtasks /Create /TN "StartDockerDesktop" /RU "SYSTEM" /SC ONSTART /TR "C:\\Program Files\\Docker\\Docker\\Docker Desktop.exe" /F',
    unless  => 'schtasks /Query /TN "StartDockerDesktop"',
    provider => powershell,
  }

}
