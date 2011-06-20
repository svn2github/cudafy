namespace CudafyModuleViewer
{
    partial class MainForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(MainForm));
            this.menuStrip1 = new System.Windows.Forms.MenuStrip();
            this.miFile = new System.Windows.Forms.ToolStripMenuItem();
            this.miOpen = new System.Windows.Forms.ToolStripMenuItem();
            this.miSaveAs = new System.Windows.Forms.ToolStripMenuItem();
            this.miExit = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripMenuItem1 = new System.Windows.Forms.ToolStripMenuItem();
            this.miEnableEditing = new System.Windows.Forms.ToolStripMenuItem();
            this.miHelp = new System.Windows.Forms.ToolStripMenuItem();
            this.miAbout = new System.Windows.Forms.ToolStripMenuItem();
            this.openFileDialog = new System.Windows.Forms.OpenFileDialog();
            this.tabControl1 = new System.Windows.Forms.TabControl();
            this.tpFunctions = new System.Windows.Forms.TabPage();
            this.tbFunctions = new System.Windows.Forms.TextBox();
            this.lbFunctions = new System.Windows.Forms.ListBox();
            this.tpTypes = new System.Windows.Forms.TabPage();
            this.tbTypes = new System.Windows.Forms.TextBox();
            this.lbTypes = new System.Windows.Forms.ListBox();
            this.tpConstants = new System.Windows.Forms.TabPage();
            this.tbConstants = new System.Windows.Forms.TextBox();
            this.lbConstants = new System.Windows.Forms.ListBox();
            this.tpSource = new System.Windows.Forms.TabPage();
            this.gbCompile = new System.Windows.Forms.GroupBox();
            this.cbArch = new System.Windows.Forms.ComboBox();
            this.btnCompile = new System.Windows.Forms.Button();
            this.label3 = new System.Windows.Forms.Label();
            this.tbSource = new System.Windows.Forms.TextBox();
            this.tpPTX = new System.Windows.Forms.TabPage();
            this.tbPTX = new System.Windows.Forms.TextBox();
            this.saveFileDialog = new System.Windows.Forms.SaveFileDialog();
            this.lbPTX = new System.Windows.Forms.ListBox();
            this.cb32bit = new System.Windows.Forms.CheckBox();
            this.cb64bit = new System.Windows.Forms.CheckBox();
            this.menuStrip1.SuspendLayout();
            this.tabControl1.SuspendLayout();
            this.tpFunctions.SuspendLayout();
            this.tpTypes.SuspendLayout();
            this.tpConstants.SuspendLayout();
            this.tpSource.SuspendLayout();
            this.gbCompile.SuspendLayout();
            this.tpPTX.SuspendLayout();
            this.SuspendLayout();
            // 
            // menuStrip1
            // 
            this.menuStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.miFile,
            this.toolStripMenuItem1,
            this.miHelp});
            this.menuStrip1.Location = new System.Drawing.Point(0, 0);
            this.menuStrip1.Name = "menuStrip1";
            this.menuStrip1.Size = new System.Drawing.Size(568, 24);
            this.menuStrip1.TabIndex = 0;
            this.menuStrip1.Text = "menuStrip1";
            // 
            // miFile
            // 
            this.miFile.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.miOpen,
            this.miSaveAs,
            this.miExit});
            this.miFile.Name = "miFile";
            this.miFile.Size = new System.Drawing.Size(37, 20);
            this.miFile.Text = "File";
            this.miFile.DropDownOpening += new System.EventHandler(this.miFile_DropDownOpening);
            // 
            // miOpen
            // 
            this.miOpen.Name = "miOpen";
            this.miOpen.Size = new System.Drawing.Size(121, 22);
            this.miOpen.Text = "Open...";
            this.miOpen.Click += new System.EventHandler(this.miOpen_Click);
            // 
            // miSaveAs
            // 
            this.miSaveAs.Name = "miSaveAs";
            this.miSaveAs.Size = new System.Drawing.Size(121, 22);
            this.miSaveAs.Text = "Save as...";
            this.miSaveAs.Click += new System.EventHandler(this.miSaveAs_Click);
            // 
            // miExit
            // 
            this.miExit.Name = "miExit";
            this.miExit.Size = new System.Drawing.Size(121, 22);
            this.miExit.Text = "Exit";
            this.miExit.Click += new System.EventHandler(this.miExit_Click);
            // 
            // toolStripMenuItem1
            // 
            this.toolStripMenuItem1.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.miEnableEditing});
            this.toolStripMenuItem1.Name = "toolStripMenuItem1";
            this.toolStripMenuItem1.Size = new System.Drawing.Size(61, 20);
            this.toolStripMenuItem1.Text = "Options";
            // 
            // miEnableEditing
            // 
            this.miEnableEditing.CheckOnClick = true;
            this.miEnableEditing.Name = "miEnableEditing";
            this.miEnableEditing.Size = new System.Drawing.Size(149, 22);
            this.miEnableEditing.Text = "Enable Editing";
            this.miEnableEditing.CheckedChanged += new System.EventHandler(this.miEnableEditing_CheckedChanged);
            // 
            // miHelp
            // 
            this.miHelp.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.miAbout});
            this.miHelp.Name = "miHelp";
            this.miHelp.Size = new System.Drawing.Size(44, 20);
            this.miHelp.Text = "Help";
            // 
            // miAbout
            // 
            this.miAbout.Name = "miAbout";
            this.miAbout.Size = new System.Drawing.Size(116, 22);
            this.miAbout.Text = "About...";
            this.miAbout.Click += new System.EventHandler(this.miAbout_Click);
            // 
            // openFileDialog
            // 
            this.openFileDialog.Filter = "Cudafy Modules|*.cdfy";
            // 
            // tabControl1
            // 
            this.tabControl1.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.tabControl1.Controls.Add(this.tpFunctions);
            this.tabControl1.Controls.Add(this.tpTypes);
            this.tabControl1.Controls.Add(this.tpConstants);
            this.tabControl1.Controls.Add(this.tpSource);
            this.tabControl1.Controls.Add(this.tpPTX);
            this.tabControl1.Location = new System.Drawing.Point(12, 27);
            this.tabControl1.Name = "tabControl1";
            this.tabControl1.SelectedIndex = 0;
            this.tabControl1.Size = new System.Drawing.Size(543, 490);
            this.tabControl1.TabIndex = 2;
            // 
            // tpFunctions
            // 
            this.tpFunctions.Controls.Add(this.tbFunctions);
            this.tpFunctions.Controls.Add(this.lbFunctions);
            this.tpFunctions.Location = new System.Drawing.Point(4, 22);
            this.tpFunctions.Name = "tpFunctions";
            this.tpFunctions.Padding = new System.Windows.Forms.Padding(3);
            this.tpFunctions.Size = new System.Drawing.Size(535, 464);
            this.tpFunctions.TabIndex = 0;
            this.tpFunctions.Text = "Functions";
            this.tpFunctions.UseVisualStyleBackColor = true;
            // 
            // tbFunctions
            // 
            this.tbFunctions.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.tbFunctions.Font = new System.Drawing.Font("Courier New", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.tbFunctions.Location = new System.Drawing.Point(17, 139);
            this.tbFunctions.Multiline = true;
            this.tbFunctions.Name = "tbFunctions";
            this.tbFunctions.ReadOnly = true;
            this.tbFunctions.ScrollBars = System.Windows.Forms.ScrollBars.Both;
            this.tbFunctions.Size = new System.Drawing.Size(500, 262);
            this.tbFunctions.TabIndex = 6;
            this.tbFunctions.WordWrap = false;
            // 
            // lbFunctions
            // 
            this.lbFunctions.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.lbFunctions.FormattingEnabled = true;
            this.lbFunctions.Location = new System.Drawing.Point(17, 14);
            this.lbFunctions.Name = "lbFunctions";
            this.lbFunctions.Size = new System.Drawing.Size(500, 108);
            this.lbFunctions.TabIndex = 5;
            this.lbFunctions.SelectedIndexChanged += new System.EventHandler(this.lbFunctions_SelectedIndexChanged);
            // 
            // tpTypes
            // 
            this.tpTypes.Controls.Add(this.tbTypes);
            this.tpTypes.Controls.Add(this.lbTypes);
            this.tpTypes.Location = new System.Drawing.Point(4, 22);
            this.tpTypes.Name = "tpTypes";
            this.tpTypes.Padding = new System.Windows.Forms.Padding(3);
            this.tpTypes.Size = new System.Drawing.Size(535, 464);
            this.tpTypes.TabIndex = 1;
            this.tpTypes.Text = "Types";
            this.tpTypes.UseVisualStyleBackColor = true;
            // 
            // tbTypes
            // 
            this.tbTypes.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.tbTypes.Font = new System.Drawing.Font("Courier New", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.tbTypes.Location = new System.Drawing.Point(16, 142);
            this.tbTypes.Multiline = true;
            this.tbTypes.Name = "tbTypes";
            this.tbTypes.ReadOnly = true;
            this.tbTypes.ScrollBars = System.Windows.Forms.ScrollBars.Both;
            this.tbTypes.Size = new System.Drawing.Size(496, 262);
            this.tbTypes.TabIndex = 3;
            this.tbTypes.WordWrap = false;
            // 
            // lbTypes
            // 
            this.lbTypes.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.lbTypes.FormattingEnabled = true;
            this.lbTypes.Location = new System.Drawing.Point(17, 14);
            this.lbTypes.Name = "lbTypes";
            this.lbTypes.Size = new System.Drawing.Size(496, 108);
            this.lbTypes.TabIndex = 1;
            this.lbTypes.SelectedIndexChanged += new System.EventHandler(this.lbTypes_SelectedIndexChanged);
            // 
            // tpConstants
            // 
            this.tpConstants.Controls.Add(this.tbConstants);
            this.tpConstants.Controls.Add(this.lbConstants);
            this.tpConstants.Location = new System.Drawing.Point(4, 22);
            this.tpConstants.Name = "tpConstants";
            this.tpConstants.Padding = new System.Windows.Forms.Padding(3);
            this.tpConstants.Size = new System.Drawing.Size(535, 464);
            this.tpConstants.TabIndex = 2;
            this.tpConstants.Text = "Constants";
            this.tpConstants.UseVisualStyleBackColor = true;
            // 
            // tbConstants
            // 
            this.tbConstants.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.tbConstants.Font = new System.Drawing.Font("Courier New", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.tbConstants.Location = new System.Drawing.Point(20, 147);
            this.tbConstants.Multiline = true;
            this.tbConstants.Name = "tbConstants";
            this.tbConstants.ReadOnly = true;
            this.tbConstants.ScrollBars = System.Windows.Forms.ScrollBars.Both;
            this.tbConstants.Size = new System.Drawing.Size(495, 262);
            this.tbConstants.TabIndex = 5;
            this.tbConstants.WordWrap = false;
            // 
            // lbConstants
            // 
            this.lbConstants.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.lbConstants.FormattingEnabled = true;
            this.lbConstants.Location = new System.Drawing.Point(17, 14);
            this.lbConstants.Name = "lbConstants";
            this.lbConstants.Size = new System.Drawing.Size(496, 108);
            this.lbConstants.TabIndex = 4;
            this.lbConstants.SelectedIndexChanged += new System.EventHandler(this.lbConstants_SelectedIndexChanged);
            // 
            // tpSource
            // 
            this.tpSource.Controls.Add(this.gbCompile);
            this.tpSource.Controls.Add(this.tbSource);
            this.tpSource.Location = new System.Drawing.Point(4, 22);
            this.tpSource.Name = "tpSource";
            this.tpSource.Padding = new System.Windows.Forms.Padding(3);
            this.tpSource.Size = new System.Drawing.Size(535, 464);
            this.tpSource.TabIndex = 3;
            this.tpSource.Text = "Generated Source";
            this.tpSource.UseVisualStyleBackColor = true;
            // 
            // gbCompile
            // 
            this.gbCompile.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Bottom | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.gbCompile.Controls.Add(this.cb64bit);
            this.gbCompile.Controls.Add(this.cb32bit);
            this.gbCompile.Controls.Add(this.cbArch);
            this.gbCompile.Controls.Add(this.btnCompile);
            this.gbCompile.Controls.Add(this.label3);
            this.gbCompile.Enabled = false;
            this.gbCompile.Location = new System.Drawing.Point(17, 361);
            this.gbCompile.Name = "gbCompile";
            this.gbCompile.Size = new System.Drawing.Size(495, 66);
            this.gbCompile.TabIndex = 21;
            this.gbCompile.TabStop = false;
            this.gbCompile.Text = "Compile";
            // 
            // cbArch
            // 
            this.cbArch.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.cbArch.Location = new System.Drawing.Point(266, 25);
            this.cbArch.Name = "cbArch";
            this.cbArch.Size = new System.Drawing.Size(125, 21);
            this.cbArch.TabIndex = 20;
            // 
            // btnCompile
            // 
            this.btnCompile.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.btnCompile.Location = new System.Drawing.Point(412, 22);
            this.btnCompile.Name = "btnCompile";
            this.btnCompile.Size = new System.Drawing.Size(75, 27);
            this.btnCompile.TabIndex = 7;
            this.btnCompile.Text = "Compile";
            this.btnCompile.UseVisualStyleBackColor = true;
            this.btnCompile.Click += new System.EventHandler(this.btnCompile_Click);
            // 
            // label3
            // 
            this.label3.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.label3.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.label3.Location = new System.Drawing.Point(194, 29);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(66, 15);
            this.label3.TabIndex = 17;
            this.label3.Text = "Architecture:";
            // 
            // tbSource
            // 
            this.tbSource.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.tbSource.Font = new System.Drawing.Font("Courier New", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.tbSource.Location = new System.Drawing.Point(17, 14);
            this.tbSource.Multiline = true;
            this.tbSource.Name = "tbSource";
            this.tbSource.ReadOnly = true;
            this.tbSource.ScrollBars = System.Windows.Forms.ScrollBars.Both;
            this.tbSource.Size = new System.Drawing.Size(495, 341);
            this.tbSource.TabIndex = 6;
            this.tbSource.WordWrap = false;
            // 
            // tpPTX
            // 
            this.tpPTX.Controls.Add(this.lbPTX);
            this.tpPTX.Controls.Add(this.tbPTX);
            this.tpPTX.Location = new System.Drawing.Point(4, 22);
            this.tpPTX.Name = "tpPTX";
            this.tpPTX.Padding = new System.Windows.Forms.Padding(3);
            this.tpPTX.Size = new System.Drawing.Size(535, 464);
            this.tpPTX.TabIndex = 4;
            this.tpPTX.Text = "PTX";
            this.tpPTX.UseVisualStyleBackColor = true;
            // 
            // tbPTX
            // 
            this.tbPTX.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.tbPTX.Font = new System.Drawing.Font("Courier New", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.tbPTX.Location = new System.Drawing.Point(17, 145);
            this.tbPTX.Multiline = true;
            this.tbPTX.Name = "tbPTX";
            this.tbPTX.ReadOnly = true;
            this.tbPTX.ScrollBars = System.Windows.Forms.ScrollBars.Both;
            this.tbPTX.Size = new System.Drawing.Size(495, 291);
            this.tbPTX.TabIndex = 7;
            this.tbPTX.WordWrap = false;
            // 
            // saveFileDialog
            // 
            this.saveFileDialog.Filter = "Cudafy Module|*.cdfy";
            // 
            // lbPTX
            // 
            this.lbPTX.Anchor = ((System.Windows.Forms.AnchorStyles)(((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.lbPTX.FormattingEnabled = true;
            this.lbPTX.Location = new System.Drawing.Point(17, 14);
            this.lbPTX.Name = "lbPTX";
            this.lbPTX.Size = new System.Drawing.Size(496, 108);
            this.lbPTX.TabIndex = 8;
            this.lbPTX.SelectedIndexChanged += new System.EventHandler(this.lbPTX_SelectedIndexChanged);
            // 
            // cb32bit
            // 
            this.cb32bit.AutoSize = true;
            this.cb32bit.Checked = true;
            this.cb32bit.CheckState = System.Windows.Forms.CheckState.Checked;
            this.cb32bit.Location = new System.Drawing.Point(14, 27);
            this.cb32bit.Name = "cb32bit";
            this.cb32bit.Size = new System.Drawing.Size(52, 17);
            this.cb32bit.TabIndex = 21;
            this.cb32bit.Text = "32-bit";
            this.cb32bit.UseVisualStyleBackColor = true;
            // 
            // cb64bit
            // 
            this.cb64bit.AutoSize = true;
            this.cb64bit.Checked = true;
            this.cb64bit.CheckState = System.Windows.Forms.CheckState.Checked;
            this.cb64bit.Location = new System.Drawing.Point(71, 27);
            this.cb64bit.Name = "cb64bit";
            this.cb64bit.Size = new System.Drawing.Size(52, 17);
            this.cb64bit.TabIndex = 21;
            this.cb64bit.Text = "64-bit";
            this.cb64bit.UseVisualStyleBackColor = true;
            // 
            // MainForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(568, 524);
            this.Controls.Add(this.menuStrip1);
            this.Controls.Add(this.tabControl1);
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.MainMenuStrip = this.menuStrip1;
            this.Name = "MainForm";
            this.Text = "Cudafy Module Viewer - Hybrid DSP Systems";
            this.menuStrip1.ResumeLayout(false);
            this.menuStrip1.PerformLayout();
            this.tabControl1.ResumeLayout(false);
            this.tpFunctions.ResumeLayout(false);
            this.tpFunctions.PerformLayout();
            this.tpTypes.ResumeLayout(false);
            this.tpTypes.PerformLayout();
            this.tpConstants.ResumeLayout(false);
            this.tpConstants.PerformLayout();
            this.tpSource.ResumeLayout(false);
            this.tpSource.PerformLayout();
            this.gbCompile.ResumeLayout(false);
            this.gbCompile.PerformLayout();
            this.tpPTX.ResumeLayout(false);
            this.tpPTX.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.MenuStrip menuStrip1;
        private System.Windows.Forms.ToolStripMenuItem miFile;
        private System.Windows.Forms.ToolStripMenuItem miOpen;
        private System.Windows.Forms.ToolStripMenuItem miExit;
        private System.Windows.Forms.ToolStripMenuItem miHelp;
        private System.Windows.Forms.ToolStripMenuItem miAbout;
        private System.Windows.Forms.OpenFileDialog openFileDialog;
        private System.Windows.Forms.TabControl tabControl1;
        private System.Windows.Forms.TabPage tpFunctions;
        private System.Windows.Forms.TabPage tpTypes;
        private System.Windows.Forms.TabPage tpConstants;
        private System.Windows.Forms.TabPage tpSource;
        private System.Windows.Forms.TabPage tpPTX;
        private System.Windows.Forms.TextBox tbFunctions;
        private System.Windows.Forms.ListBox lbFunctions;
        private System.Windows.Forms.TextBox tbTypes;
        private System.Windows.Forms.ListBox lbTypes;
        private System.Windows.Forms.TextBox tbConstants;
        private System.Windows.Forms.ListBox lbConstants;
        private System.Windows.Forms.TextBox tbSource;
        private System.Windows.Forms.TextBox tbPTX;
        private System.Windows.Forms.ToolStripMenuItem toolStripMenuItem1;
        private System.Windows.Forms.ToolStripMenuItem miEnableEditing;
        private System.Windows.Forms.ToolStripMenuItem miSaveAs;
        private System.Windows.Forms.Button btnCompile;
        private System.Windows.Forms.SaveFileDialog saveFileDialog;
        private System.Windows.Forms.GroupBox gbCompile;
        private System.Windows.Forms.ComboBox cbArch;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.ListBox lbPTX;
        private System.Windows.Forms.CheckBox cb64bit;
        private System.Windows.Forms.CheckBox cb32bit;
    }
}

