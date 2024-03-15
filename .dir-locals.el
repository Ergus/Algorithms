((nil . ((eglot-workspace-configuration
	  . (:clangd (:initializationOptions (:compilationDatabasePath "build"))))) ;; compile_commands.json is in /build
      )
 (auto-mode-alist . (("/cuda/[^/]+\\.hpp\\'" . cuda-mode))) ;; cuda/*.hpp in cuda mode
)
