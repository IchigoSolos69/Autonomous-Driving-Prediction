Unicode true
!include "MUI2.nsh"
!include "LogicLib.nsh"
!include "x64.nsh"
!include "WinVer.nsh"
!ifndef APP_VERSION
!define APP_VERSION "0.1.0-beta"
!endif
!ifndef APP_EDITION
!define APP_EDITION "cpu"
!endif
Name "Interaction Lab (${APP_EDITION} research beta)"
OutFile "${OUTPUT_FILE}"
InstallDir "$LOCALAPPDATA\Programs\Interaction Lab"
InstallDirRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\InteractionLab" "InstallLocation"
RequestExecutionLevel user
SetCompressor /SOLID lzma
SetCompressorDictSize 64
ShowInstDetails show
!define MUI_ABORTWARNING
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

Function .onInit
    ${IfNot} ${RunningX64}
        MessageBox MB_OK|MB_ICONSTOP "Interaction Lab requires 64-bit Windows 11." /SD IDOK
        Abort
    ${EndIf}
    ReadRegStr $0 HKLM "SOFTWARE\Microsoft\Windows NT\CurrentVersion" "CurrentBuildNumber"
    ${If} $0 < 22000
        MessageBox MB_OK|MB_ICONSTOP "This private beta supports Windows 11 or newer." /SD IDOK
        Abort
    ${EndIf}
    System::Call 'kernel32::OpenMutexW(i 0x100000, i 0, w "Local\InteractionLab.ResearchBeta") p.r0'
    ${If} $0 <> 0
        System::Call 'kernel32::CloseHandle(p r0)'
        MessageBox MB_OK|MB_ICONSTOP "Close Interaction Lab before installing or upgrading it." /SD IDOK
        Abort
    ${EndIf}
FunctionEnd

Section "Application" SEC_APP
    SetShellVarContext current
    StrCpy $1 ""
    ReadRegStr $1 HKCU "Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" "pv"
    ${If} $1 == ""
        SetRegView 32
        ReadRegStr $1 HKLM "Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" "pv"
    ${EndIf}
    ${If} $1 == ""
        SetRegView 64
        ReadRegStr $1 HKLM "Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" "pv"
    ${EndIf}
    ${If} $1 == ""
    ${OrIf} $1 == "0.0.0.0"
        InitPluginsDir
        SetOutPath "$PLUGINSDIR"
        File /oname=MicrosoftEdgeWebview2Setup.exe "${WEBVIEW_BOOTSTRAPPER}"
        ExecWait '"$PLUGINSDIR\MicrosoftEdgeWebview2Setup.exe" /silent /install' $0
        ${If} $0 != 0
            MessageBox MB_OK|MB_ICONSTOP "Microsoft WebView2 could not be installed. Connect to the internet or install the offline Evergreen Runtime from Microsoft, then run setup again." /SD IDOK
            Abort
        ${EndIf}
    ${EndIf}
    SetOutPath "$INSTDIR"
    !ifdef APP_ARCHIVE_NAME
    InitPluginsDir
    SetOutPath "$PLUGINSDIR"
    File /oname=install-payload.ps1 "${PAYLOAD_SCRIPT}"
    DetailPrint "Verifying and preparing inference files. This can take several minutes."
    nsExec::ExecToStack '"$WINDIR\Sysnative\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -NonInteractive -File "$PLUGINSDIR\install-payload.ps1" -Archive "$EXEDIR\${APP_ARCHIVE_NAME}" -ExpectedSha "${APP_ARCHIVE_SHA256}" -Destination "$INSTDIR"'
    Pop $0
    Pop $1
    DetailPrint "$1"
    FileOpen $2 "$TEMP\InteractionLab-install.log" w
    FileWrite $2 "Payload exit: $0$\r$\n$1$\r$\n"
    FileClose $2
    ${If} $0 != 0
        MessageBox MB_OK|MB_ICONSTOP "The application payload could not be installed. Check $TEMP\InteractionLab-install.log for a missing ZIP, hash mismatch, disk space, or file access error. Existing user data was preserved." /SD IDOK
        Abort
    ${EndIf}
    SetOutPath "$INSTDIR"
    !else
    File /r "${APP_DIRECTORY}\*"
    !endif
    WriteUninstaller "$INSTDIR\Uninstall.exe"
    CreateDirectory "$SMPROGRAMS\Interaction Lab"
    CreateShortcut "$SMPROGRAMS\Interaction Lab\Interaction Lab.lnk" "$INSTDIR\Interaction Lab.exe"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\InteractionLab" "DisplayName" "Interaction Lab (${APP_EDITION} research beta)"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\InteractionLab" "DisplayVersion" "${APP_VERSION}"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\InteractionLab" "InstallLocation" "$INSTDIR"
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\InteractionLab" "UninstallString" '"$INSTDIR\Uninstall.exe"'
    WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\InteractionLab" "NoModify" 1
    WriteRegDWORD HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\InteractionLab" "NoRepair" 1
SectionEnd

Section "Uninstall"
    SetShellVarContext current
    System::Call 'kernel32::OpenMutexW(i 0x100000, i 0, w "Local\InteractionLab.ResearchBeta") p.r0'
    ${If} $0 <> 0
        System::Call 'kernel32::CloseHandle(p r0)'
        MessageBox MB_OK|MB_ICONSTOP "Close Interaction Lab before uninstalling it." /SD IDOK
        Abort
    ${EndIf}
    Delete "$SMPROGRAMS\Interaction Lab\Interaction Lab.lnk"
    RMDir "$SMPROGRAMS\Interaction Lab"
    DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\InteractionLab"
    # User datasets/models/logs are outside INSTDIR and deliberately retained.
    RMDir /r "$INSTDIR\_internal"
    Delete "$INSTDIR\Interaction Lab.exe"
    Delete "$INSTDIR\Uninstall.exe"
    RMDir "$INSTDIR"
SectionEnd
